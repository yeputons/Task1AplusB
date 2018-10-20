#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>
#include <optional>
#include <utility>
#include <string_view>
#include <string>

using std::optional;
using std::nullopt;
using std::make_optional;
using std::vector;
using std::runtime_error;
using std::string;
using std::string_view;

template <typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

optional<cl_device_id> findDevice(cl_device_type required_type) {
    cl_uint platformsCount;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    assert(platformsCount > 0);

    vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (const auto &platform : platforms) {
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (const auto &deviceId : devices) {
            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof (deviceType), &deviceType, nullptr));

            if (deviceType == required_type) {
                return make_optional(deviceId);
            }
        }
    }
    return nullopt;
}

template<typename Descriptor>
class ClHolder : public Descriptor {
public:

    ClHolder() {
    }

    template<typename... Args>
    ClHolder(Args&&... args) {
        cl_int errcode;
        value_ = Descriptor::create(std::forward<Args>(args)..., &errcode);
        OCL_SAFE_CALL(errcode);
    }

    ~ClHolder() {
        if (value_.has_value()) {
            Descriptor::release(value_.value());
        }
    }

    ClHolder(const ClHolder&) = delete;

    ClHolder(ClHolder &&other) {
        value_ = other.value_;
        other.value_.reset();
    }

    typename Descriptor::T get() {
        return value_.value();
    }

private:
    optional<typename Descriptor::T> value_;
};

struct ClContextDescriptor {
    using T = cl_context;

    static T create(cl_device_id deviceId, cl_int *errcode) {
        return clCreateContext(
                nullptr,
                1,
                &deviceId,
                nullptr,
                nullptr,
                errcode);
    }

    static void release(T context) {
        clReleaseContext(context);
    }
};

using ClContext = ClHolder<ClContextDescriptor>;

struct ClCommandInOrderQueueDescriptor {
    using T = cl_command_queue;

    static T create(ClContext &context, cl_device_id device, cl_int *errcode) {
        return clCreateCommandQueue(context.get(), device, 0, errcode);
    }

    static void release(T queue) {
        clReleaseCommandQueue(queue);
    }
};

using ClCommandInOrderQueue = ClHolder<ClCommandInOrderQueueDescriptor>;

struct ClBufferDescriptor {
    using T = cl_mem;

    static T create(ClContext &context, cl_mem_flags flags, size_t size, void *data, cl_int *errcode) {
        return clCreateBuffer(context.get(), flags, size, data, errcode);
    }

    static void release(T buffer) {
        clReleaseMemObject(buffer);
    }
};

using ClBuffer = ClHolder<ClBufferDescriptor>;

struct ClProgramDescriptor {
    using T = cl_program;

    static T create(ClContext &context, string_view source, cl_int *errcode) {
        const char *strs[] = {source.data()};
        const size_t lens[] = {source.length()};
        return clCreateProgramWithSource(context.get(), 1, strs, lens, errcode);
    }

    static void release(T program) {
        clReleaseProgram(program);
    }
};

using ClProgram = ClHolder<ClProgramDescriptor>;

struct ClKernelDescriptor {
    using T = cl_kernel;

    static T create(ClProgram &program, const char *kernel_name, cl_int *errcode) {
        return clCreateKernel(program.get(), kernel_name, errcode);
    }

    static void release(T kernel) {
        clReleaseKernel(kernel);
    }
};

class ClKernel : public ClHolder<ClKernelDescriptor> {
public:

    ClKernel(ClProgram &program, const char *kernel_name) : ClHolder<ClKernelDescriptor>(program, kernel_name) {
    }

    template<typename TArg>
    void setArg(cl_uint arg_index, TArg arg) {
        OCL_SAFE_CALL(clSetKernelArg(get(), arg_index, sizeof (arg), &arg));
    }
};

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с заданием Example0EnumDevices узнайте какие есть устройства, и выберите из них какое-нибудь
    // (если есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_device_id device;
    {
        optional<cl_device_id> found_device;
        if (!found_device) found_device = findDevice(CL_DEVICE_TYPE_GPU);
        if (!found_device) found_device = findDevice(CL_DEVICE_TYPE_CPU);
        if (!found_device) throw runtime_error("Unable to find suitable device");
        device = found_device.value();
    }

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    ClContext context(device);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue
    ClCommandInOrderQueue queue(context, device);

    unsigned int n = 10 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт что чисел в каждом массиве - n штук
    // Данные в as и bs можно прогрузить этим же методом скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    static_assert(sizeof (float) == 4);
    const auto BUFFER_SIZE = sizeof (float) * n;
    ClBuffer as_gpu(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, BUFFER_SIZE, as.data());
    ClBuffer bs_gpu(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, BUFFER_SIZE, bs.data());
    ClBuffer cs_gpu(context, CL_MEM_READ_ONLY, BUFFER_SIZE, nullptr);

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания)
    // напечатав исходники в консоль (if проверяет что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание что передать вам нужно указатель на указатель
    ClProgram program(context, kernel_sources);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram

    // А так же напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // см. clGetProgramBuildInfo
    //    size_t log_size = 0;
    //    std::vector<char> log(log_size, 0);
    //    if (log_size > 1) {
    //        std::cout << "Log:" << std::endl;
    //        std::cout << log.data() << std::endl;
    //    }
    {
        cl_uint build_result = clBuildProgram(program.get(), 0, nullptr, nullptr, nullptr, nullptr);
        size_t log_size;
        OCL_SAFE_CALL(clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
        vector<char> log(log_size);
        OCL_SAFE_CALL(clGetProgramBuildInfo(program.get(), device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr));
        if (!log.empty()) {
            std::cout << string(log.begin(), log.end()) << std::endl;
        }
        OCL_SAFE_CALL(build_result);
    }

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    ClKernel kernel(program, "aplusb");


    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        kernel.setArg(i++, as_gpu.get());
        kernel.setArg(i++, bs_gpu.get());
        kernel.setArg(i++, cs_gpu.get());
        kernel.setArg(i++, n);
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание что чтобы дождаться окончания вычислений (чтобы знать когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            const size_t sizes[] = {n};
            const size_t local_sizes[] = {workGroupSize};
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue.get(), kernel.get(), 1, nullptr, sizes, local_sizes, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считаются не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклониение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще) достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << static_cast<double>(n) / t.lapAvg() / 1e9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти т.о. 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3.0 * n * sizeof (float) / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue.get(), cs_gpu.get(), CL_TRUE, 0, BUFFER_SIZE, cs.data(), 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << static_cast<double>(BUFFER_SIZE) / t.lapAvg() / 1e9 << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (fabs(cs[i] - (as[i] + bs[i])) > 1e-4) {
            throw std::runtime_error("CPU and GPU results differ for i=" + to_string(i) + ": " + to_string(cs[i]) + " != " + to_string(as[i]) + " + " + to_string(bs[i]));
        }
    }

    return 0;
}
