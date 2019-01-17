#include <iostream>
#include <cmath>
#include <omp.h>
#include <limits>

double formula(double x);
double integrate(double left, double right, int segments, double step);
double altIntegrate(double left, double right, int segments, double step);
double original(double left, double right, int segments, double step);
double integrateBySections(double left, double right, int segments, double step);

struct m_time {
    double min = std::numeric_limits<double>::max();
    double average = 0;
    double max = 0;
};

int main() {

    const int SEGMENTS = 10000000; // should be even number, also named as a 2n
    const double LEFT = -3;
    const double RIGHT = 12;
    const double STEP = (RIGHT - LEFT) / SEGMENTS;
    const int RUNS = 50;

    std::cout << "Function: atan(x) - 0.2*x" << std::endl;
    std::cout << "Segments: " << SEGMENTS << std::endl;
    std::cout << "Left: " << LEFT << std::endl;
    std::cout << "Right: " << RIGHT << std::endl << std::endl;

    double start1 = omp_get_wtime();
    double value = original(LEFT, RIGHT, SEGMENTS, STEP);
    double end1 = omp_get_wtime();
    std::cout << "Serial version. Value: " << value << " : time = " << end1-start1<< std::endl;

    double start = omp_get_wtime();
    value = integrate(LEFT, RIGHT, SEGMENTS, STEP);
    double end = omp_get_wtime();

    std::cout << "Version 1. Value: " << value << " : time = " << end-start<< std::endl;

    start = omp_get_wtime();
    value = integrateBySections(LEFT, RIGHT, SEGMENTS, STEP);
    end = omp_get_wtime();

    std::cout << "Version 2 (sections). Value: " << value << " : time = " << end-start<< std::endl << std::endl;

    double time;
    int count = 0;
    double res = 0;
    m_time sTime;
    for(int i = 1; i<RUNS; ++i) {
        start = omp_get_wtime();
        value = original(LEFT, RIGHT, SEGMENTS, STEP);
        end = omp_get_wtime();
        time = end-start;
        sTime.min = sTime.min < time ? sTime.min : time;
        sTime.max = sTime.max > time ? sTime.max : time;
        sTime.average += end-start;
        count++;
    }
    sTime.average /= count;
    std::cout << "Serial version. Min time " << sTime.min<< std::endl;
    std::cout << "Serial version. Max time " << sTime.max<< std::endl;
    std::cout << "Serial version. Average time " << sTime.average<< std::endl;

    count = 0;
    res = 0;
    m_time forTime;
    for(int i = 1; i<RUNS; ++i) {
        start = omp_get_wtime();
        value = integrate(LEFT, RIGHT, SEGMENTS, STEP);
        end = omp_get_wtime();
        time = end-start;
        forTime.min = forTime.min < time ? forTime.min : time;
        forTime.max = forTime.max > time ? forTime.max : time;
        forTime.average += end-start;
        count++;
    }
    forTime.average /= count;
    std::cout << "Version 1. Min time " << forTime.min<< std::endl;
    std::cout << "Version 1. Max time " << forTime.max<< std::endl;
    std::cout << "Version 1. Average time " << forTime.average<< std::endl;
    count = 0;
    res = 0;
    m_time sectionsTime;
    for(int i = 1; i<RUNS; ++i) {
        start = omp_get_wtime();
        value = integrateBySections(LEFT, RIGHT, SEGMENTS, STEP);
        end = omp_get_wtime();
        time = end-start;
        sectionsTime.min = sectionsTime.min < time ? sectionsTime.min : time;
        sectionsTime.max = sectionsTime.max > time ? sectionsTime.max : time;
        sectionsTime.average += end-start;
        count++;
    }
    sectionsTime.average /= count;
    std::cout << "Version 2 (sections). Min time " << sectionsTime.min<< std::endl;
    std::cout << "Version 2 (sections). Max time " << sectionsTime.max<< std::endl;
    std::cout << "Version 2 (sections). Average time " << sectionsTime.average<< std::endl;

    return 0;
}

double formula(double x) {
    return atan(x) - 0.2 * x;
}

double integrate(double left, double right, int segments, double step) {
    double evenSegments = 0;
    double oddSegments = 0;

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

double altIntegrate(double left, double right, int segments, double step) {
    double evenSegments = 0;
    double oddSegments = 0;
    #pragma omp parallel for reduction(+: evenSegments, oddSegments) num_threads(2)
    for (int i = 1; i < segments; i++ ) {
        i%2 == 0 ? evenSegments += formula(left + step * i) : oddSegments += formula(left + step * i);
    }
    evenSegments *= 2;
    oddSegments *= 4;

    return (step / 3) * (formula(left) + formula(right) + evenSegments + oddSegments);
}

double original(double left, double right, int segments, double step) {
    double evenSegments = 0;
    double oddSegments = 0;

    for (int i = 1; i < segments; i++ ) {
        i%2 == 0 ? evenSegments += formula(left + step * i) : oddSegments += formula(left + step * i);
    }
    evenSegments *= 2;
    oddSegments *= 4;

    return (step / 3) * (formula(left) + formula(right) + evenSegments + oddSegments);
}

double integrateBySections(double left, double right, int segments, double step) {
    double evenSegments = 0;
    double oddSegments = 0;

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
