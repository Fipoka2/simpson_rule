#include <iostream>
#include <cmath>
#include <omp.h>

float formula(float x);
float integrate(float left, float right, int segments, float step);
float altIntegrate(float left, float right, int segments, float step);
float original(float left, float right, int segments, float step);
float integrateBySections(float left, float right, int segments, float step);

int main() {

    const int SEGMENTS = 2000; // should be even number, also named as a 2n
    const float LEFT = -3;
    const float RIGHT = 12;
    const float STEP = (RIGHT - LEFT) / SEGMENTS;

    float start1 = omp_get_wtime();
    float value = original(LEFT, RIGHT, SEGMENTS, STEP);
    float end1 = omp_get_wtime();

    std::cout << "Последовательная версия. Результат: " << value << " : time = " << end1-start1<< std::endl;

    float start = omp_get_wtime();
    value = integrate(LEFT, RIGHT, SEGMENTS, STEP);
    float end = omp_get_wtime();

    std::cout << "Версия 1. Результат: " << value << " : time = " << end-start<< std::endl;

    start = omp_get_wtime();
    value = integrateBySections(LEFT, RIGHT, SEGMENTS, STEP);
    end = omp_get_wtime();

    std::cout << "Версия 2 (секции). Результат: " << value << " : time = " << end-start<< std::endl;

    int count = 0;
    float res = 0;

    for(int i = 1; i<50; ++i) {
        start = omp_get_wtime();
        value = original(LEFT, RIGHT, SEGMENTS, STEP);
        end = omp_get_wtime();
        res += end-start;
        count++;
    }
    std::cout << "Последовательная версия.Среднее время " << res/count<< std::endl;

    count = 0;
    res = 0;
    for(int i = 1; i<50; ++i) {
        start = omp_get_wtime();
        value = integrate(LEFT, RIGHT, SEGMENTS, STEP);
        end = omp_get_wtime();
        res += end-start;
        count++;
    }
    std::cout << "Версия 1.Среднее время " << res/count<< std::endl;

    count = 0;
    res = 0;
    for(int i = 1; i<50; ++i) {
        start = omp_get_wtime();
        value = integrateBySections(LEFT, RIGHT, SEGMENTS, STEP);
        end = omp_get_wtime();
        res += end-start;
        count++;
    }
    std::cout << "Версия 2 (секции).Среднее время " << res/count<< std::endl;


    return 0;
}

float formula(float x) {
    return atan(x) - 0.2 * x;
}

float integrate(float left, float right, int segments, float step) {
    float evenSegments = 0;
    float oddSegments = 0;

    #pragma omp parallel for reduction(+: evenSegments) num_threads(8)
    for (int i = 2; i < segments; i += 2 ) {
        evenSegments += formula(left + step*i);
    }
    evenSegments *= 2;

    #pragma omp parallel for reduction(+: oddSegments) num_threads(8)
    for (int i = 1; i < segments; i += 2 ) {
        oddSegments += formula(left + step * i);
    }
    oddSegments *= 4;

    return (step / 3) * (formula(left) + formula(right) + evenSegments + oddSegments);
}

float altIntegrate(float left, float right, int segments, float step) {
    float evenSegments = 0;
    float oddSegments = 0;
    #pragma omp parallel for reduction(+: evenSegments, oddSegments) num_threads(8)
    for (int i = 1; i < segments; i++ ) {
        i%2 == 0 ? evenSegments += formula(left + step * i) : oddSegments += formula(left + step * i);
    }
    evenSegments *= 2;
    oddSegments *= 4;

    return (step / 3) * (formula(left) + formula(right) + evenSegments + oddSegments);
}

float original(float left, float right, int segments, float step) {
    float evenSegments = 0;
    float oddSegments = 0;

    for (int i = 1; i < segments; i++ ) {
        i%2 == 0 ? evenSegments += formula(left + step * i) : oddSegments += formula(left + step * i);
    }
    evenSegments *= 2;
    oddSegments *= 4;

    return (step / 3) * (formula(left) + formula(right) + evenSegments + oddSegments);
}

float integrateBySections(float left, float right, int segments, float step) {
    float evenSegments = 0;
    float oddSegments = 0;

    #pragma omp parallel reduction(+:evenSegments, oddSegments)
    {
        #pragma omp sections
        {
            #pragma  omp section
            {
                // нечётные до середины
                for (int i = 1; i < segments / 2; i += 2 ) {
                    oddSegments += formula(left + step * i);
                }

            }
            #pragma  omp section
            {

                // нечётные с середины
                for (int i = segments / 2 + 1; i < segments; i += 2 ) {

                    oddSegments += formula(left + step * i);
                }

            }
            #pragma  omp section
            {
                // чётные до середины
                for (int i = 2; i < segments / 2; i += 2 ) {
                    evenSegments += formula(left + step * i);
                }

            }
            #pragma  omp section
            {
                // чётные с середины
                for (int i = segments / 2; i < segments; i += 2 ) {
                    evenSegments += formula(left + step * i);
                }

            }

        }
    };
    evenSegments *= 2;
    oddSegments *= 4;

    return (step / 3) * (formula(left) + formula(right) + evenSegments + oddSegments);
}
