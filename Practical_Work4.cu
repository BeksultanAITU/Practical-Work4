#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)


// Параметры "демо-алгоритмов"

static constexpr int BLOCK_SIZE = 256;

// Для сортировки: пузырьком сортируем только маленькие чанки,
// иначе пузырёк будет слишком медленный.
static constexpr int CHUNK_SIZE = 64;

// Для merge через shared: ограничим seg, чтобы не превышать shared memory.
static constexpr int MAX_SHARED_SEG = 1024;


// TASK 2a: Редукция суммы "только глобальная память"
// каждый поток суммирует часть массива и делает atomicAdd в один глобальный аккумулятор.

__global__ void sum_global_atomic(const float* __restrict__ in, float* outSum, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float local = 0.0f; // временная сумма потока (обычно регистр)

    // Stride-обход: чтобы покрыть весь массив даже если потоков меньше, чем элементов
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        local += in[i]; // чтение из global
    }

    // Сведение результатов в один глобальный аккумулятор
    atomicAdd(outSum, local);
}


// TASK 2b: Редукция суммы "глобальная + shared"
// в каждом блоке загружаем в shared и делаем редукцию внутри блока,
// затем пишем по одному значению на блок в global. Повторяем проходы до 1 значения.

__global__ void reduce_block_shared(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    // Загружаем в shared (если вышли за границу массива — кладём 0)
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads(); // обязательно: все потоки должны заполнить shared

    // Редукция пополам: 256 -> 128 -> 64 -> ...
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads(); // синхронизация шагов редукции
    }

    // Поток 0 записывает сумму блока в global
    if (tid == 0) out[blockIdx.x] = sdata[0];
}


// TASK 3: Сортировка
// 1) Bubble sort для чанков: данные читаем из global, сортируем в локальном массиве потока,
//    затем пишем обратно в global.
// 2) Merge: слияние отсортированных чанков, используя shared как буфер (пока seg небольшой).

__device__ void bubble_sort_local(float* a, int len) {
    for (int i = 0; i < len - 1; ++i) {
        for (int j = 0; j < len - 1 - i; ++j) {
            if (a[j] > a[j + 1]) {
                float t = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }
        }
    }
}

__global__ void bubble_sort_chunks(float* data, int n) {
    int chunk_id = blockIdx.x; // один блок = один чанк
    int start = chunk_id * CHUNK_SIZE;
    if (start >= n) return;

    // Для простоты: сортирует один поток в блоке (thread 0)
    if (threadIdx.x == 0) {
        float local[CHUNK_SIZE]; // локальный массив у потока

        int len = CHUNK_SIZE;
        if (start + len > n) len = n - start;

        // global -> local
        for (int i = 0; i < len; ++i) local[i] = data[start + i];

        // сортировка в локальной памяти
        bubble_sort_local(local, len);

        // local -> global
        for (int i = 0; i < len; ++i) data[start + i] = local[i];
    }
}

// merge через shared: используем shared как быстрый буфер для двух отсортированных частей
__global__ void merge_pass_shared(const float* __restrict__ in, float* __restrict__ out, int n, int seg) {
    int pair_id = blockIdx.x;
    int start = pair_id * (2 * seg);
    if (start >= n) return;

    int mid = start + seg;
    int end = start + 2 * seg;
    if (mid > n) mid = n;
    if (end > n) end = n;

    int left_len  = mid - start;
    int right_len = end - mid;

    extern __shared__ float shmem[];
    float* L = shmem;
    float* R = shmem + seg;

    // Для демонстрации (понятная логика): весь merge делает один поток
    if (threadIdx.x == 0) {
        for (int i = 0; i < left_len;  ++i) L[i] = in[start + i];
        for (int i = 0; i < right_len; ++i) R[i] = in[mid + i];

        int i = 0, j = 0, k = start;
        while (i < left_len && j < right_len) {
            out[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
        }
        while (i < left_len)  out[k++] = L[i++];
        while (j < right_len) out[k++] = R[j++];
    }
}

// merge без shared (корректный fallback для больших seg, когда shared уже не хватает)
__global__ void merge_pass_global(const float* __restrict__ in, float* __restrict__ out, int n, int seg) {
    int pair_id = blockIdx.x;
    int start = pair_id * (2 * seg);
    if (start >= n) return;

    int mid = start + seg;
    int end = start + 2 * seg;
    if (mid > n) mid = n;
    if (end > n) end = n;

    if (threadIdx.x == 0) {
        int i = start, j = mid, k = start;
        while (i < mid && j < end) {
            out[k++] = (in[i] <= in[j]) ? in[i++] : in[j++];
        }
        while (i < mid) out[k++] = in[i++];
        while (j < end) out[k++] = in[j++];
    }
}


// Таймер CUDA events

struct GpuTimer {
    cudaEvent_t start{}, stop{};
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic() { CUDA_CHECK(cudaEventRecord(start)); }
    float toc_ms() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// CPU сумма (для контроля корректности)
double cpu_sum(const std::vector<float>& a) {
    double s = 0.0;
    for (float x : a) s += x;
    return s;
}

// GPU сумма: вариант 2a (global atomic)
float gpu_sum_global_atomic(const float* d_in, int n, float& time_ms) {
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    // Простая настройка: фиксированное число блоков
    int blocks = 1024;

    GpuTimer t;
    t.tic();
    sum_global_atomic<<<blocks, BLOCK_SIZE>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    time_ms = t.toc_ms();

    float h_out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    return h_out;
}

// GPU сумма: вариант 2b (shared reduction в несколько проходов)
float gpu_sum_shared_reduce(const float* d_in, int n, float& time_ms) {
    GpuTimer t;
    t.tic();

    int cur_n = n;
    const float* cur_in = d_in;

    float* d_tmp1 = nullptr;
    float* d_tmp2 = nullptr;

    int blocks = (cur_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_tmp1, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp2, blocks * sizeof(float)));

    float* cur_out = d_tmp1;

    while (true) {
        blocks = (cur_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        reduce_block_shared<<<blocks, BLOCK_SIZE>>>(cur_in, cur_out, cur_n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (blocks <= 1) break;

        // Следующий проход: входом становится выход предыдущего
        cur_n = blocks;
        cur_in = cur_out;
        cur_out = (cur_out == d_tmp1) ? d_tmp2 : d_tmp1;
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, cur_out, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_tmp1));
    CUDA_CHECK(cudaFree(d_tmp2));

    time_ms = t.toc_ms();
    return result;
}

// GPU сортировка: bubble чанков + merge-проходы
float gpu_sort_bubble_merge(const std::vector<float>& h_in, float& time_ms, bool& is_sorted_ok) {
    int n = (int)h_in.size();

    float* d_a = nullptr;
    float* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer t;
    t.tic();

    // 1) Bubble sort для чанков (local)
    int chunk_blocks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    bubble_sort_chunks<<<chunk_blocks, 128>>>(d_a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2) Merge-проходы: размер сегмента удваивается каждый раз
    int seg = CHUNK_SIZE;
    bool ping = true; // true: in=d_a out=d_b, false: in=d_b out=d_a

    while (seg < n) {
        int pair_blocks = (n + 2 * seg - 1) / (2 * seg);

        if (seg <= MAX_SHARED_SEG) {
            // shared memory: нужно 2*seg float
            size_t sh_bytes = (size_t)(2 * seg) * sizeof(float);
            merge_pass_shared<<<pair_blocks, 128, sh_bytes>>>(ping ? d_a : d_b, ping ? d_b : d_a, n, seg);
        } else {
            // seg стал большой: корректный merge без shared
            merge_pass_global<<<pair_blocks, 128>>>(ping ? d_a : d_b, ping ? d_b : d_a, n, seg);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        ping = !ping;
        seg *= 2;
    }

    time_ms = t.toc_ms();

    // Копируем результат на host для проверки (по заданию важно показать результат)
    std::vector<float> h_out(n);
    float* d_final = ping ? d_a : d_b;
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_final, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    is_sorted_ok = std::is_sorted(h_out.begin(), h_out.end());
    return is_sorted_ok ? 1.0f : 0.0f; // просто "заглушка", чтобы не менять стиль вызова
}

int main() {
    std::cout << "Practical_Work4\n";

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n\n";

    // Размеры для замеров по заданию
    std::vector<int> sizes = {10'000, 100'000, 1'000'000};

    // Сюда собираем результаты, чтобы потом вывести аккуратными таблицами
    std::vector<double> cpu_sum_all;
    std::vector<float>  gpu_sum_global_all, gpu_sum_shared_all;
    std::vector<float>  t_global_all, t_shared_all, t_sort_all;
    std::vector<int>    sort_ok_all;

    cpu_sum_all.reserve(sizes.size());
    gpu_sum_global_all.reserve(sizes.size());
    gpu_sum_shared_all.reserve(sizes.size());
    t_global_all.reserve(sizes.size());
    t_shared_all.reserve(sizes.size());
    t_sort_all.reserve(sizes.size());
    sort_ok_all.reserve(sizes.size());

    
    // TASK 1
    
    std::cout << "TASK 1\n";
    std::cout << "Generating random arrays and CPU reference sums...\n\n";

    // Для честности: одинаковый seed, но разные N -> разные последовательности длины N
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Мы храним входные массивы, потому что потом они нужны для GPU задач
    std::vector<std::vector<float>> inputs;
    inputs.reserve(sizes.size());

    for (int N : sizes) {
        std::vector<float> h(N);
        for (int i = 0; i < N; ++i) h[i] = dist(rng);

        double s = cpu_sum(h);
        inputs.push_back(std::move(h));
        cpu_sum_all.push_back(s);
    }

    // Таблица для TASK 1
    std::cout << std::left
              << std::setw(12) << "N"
              << std::setw(20) << "CPU reference sum"
              << "\n";
    std::cout << std::string(32, '-') << "\n";

    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << std::left
                  << std::setw(12) << sizes[i]
                  << std::setw(20) << std::fixed << std::setprecision(6) << cpu_sum_all[i]
                  << "\n";
    }
    std::cout << "\n";

    
    // TASK 2
    
    std::cout << "TASK 2\n";
    std::cout << "Reduction: (a) Global only, (b) Global + Shared\n";
    std::cout << "Note: Shared reduces global memory traffic inside blocks.\n\n";

    for (size_t idx = 0; idx < sizes.size(); ++idx) {
        const int N = sizes[idx];
        const auto& h = inputs[idx];

        float* d_in = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        float t_g = 0.0f, t_s = 0.0f;
        float sum_g = gpu_sum_global_atomic(d_in, N, t_g);
        float sum_s = gpu_sum_shared_reduce(d_in, N, t_s);

        CUDA_CHECK(cudaFree(d_in));

        gpu_sum_global_all.push_back(sum_g);
        gpu_sum_shared_all.push_back(sum_s);
        t_global_all.push_back(t_g);
        t_shared_all.push_back(t_s);
    }

    // Таблица для TASK 2
    std::cout << std::left
              << std::setw(12) << "N"
              << std::setw(18) << "GPU sum global"
              << std::setw(18) << "t_global(ms)"
              << std::setw(18) << "GPU sum shared"
              << std::setw(18) << "t_shared(ms)"
              << "\n";
    std::cout << std::string(84, '-') << "\n";

    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << std::left
                  << std::setw(12) << sizes[i]
                  << std::setw(18) << std::fixed << std::setprecision(6) << gpu_sum_global_all[i]
                  << std::setw(18) << std::fixed << std::setprecision(3) << t_global_all[i]
                  << std::setw(18) << std::fixed << std::setprecision(6) << gpu_sum_shared_all[i]
                  << std::setw(18) << std::fixed << std::setprecision(3) << t_shared_all[i]
                  << "\n";
    }
    std::cout << "\n";

    
    // TASK 3
    
    std::cout << "TASK 3\n";
    std::cout << "Sorting: bubble chunks (local) + merge (shared where possible)\n\n";

    for (size_t idx = 0; idx < sizes.size(); ++idx) {
        const int N = sizes[idx];
        const auto& h = inputs[idx];

        float t_sort = 0.0f;
        bool ok = false;

        // Сортируем на GPU (для отчёта важно показать время и что сортировка корректная)
        gpu_sort_bubble_merge(h, t_sort, ok);

        t_sort_all.push_back(t_sort);
        sort_ok_all.push_back(ok ? 1 : 0);
    }

    // Таблица для TASK 3
    std::cout << std::left
              << std::setw(12) << "N"
              << std::setw(18) << "Sort time(ms)"
              << std::setw(18) << "Sorted OK"
              << "\n";
    std::cout << std::string(48, '-') << "\n";

    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << std::left
                  << std::setw(12) << sizes[i]
                  << std::setw(18) << std::fixed << std::setprecision(3) << t_sort_all[i]
                  << std::setw(18) << (sort_ok_all[i] ? "OK" : "FAILED")
                  << "\n";
    }
    std::cout << "\n";

    
    // TASK 4
    
    std::cout << "TASK 4\n";
    std::cout << "Performance measurement done for N = 10k / 100k / 1M.\n";

    return 0;
}