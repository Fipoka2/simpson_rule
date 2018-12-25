#include <iostream>
#include <cmath>
#include <omp.h>

double formula(double x);
double integrate(double left, double right, int segments, double step);
double altIntegrate(double left, double right, int segments, double step);

int main() {

    const int SEGMENTS = 1000000; // should be even number, also named as a 2n
    const double LEFT = -3;
    const double RIGHT = 12;
    const double STEP = (RIGHT - LEFT) / SEGMENTS;

    double start = omp_get_wtime();
    double value = integrate(LEFT, RIGHT, SEGMENTS, STEP);
    double end = omp_get_wtime();

    std::cout << "Hello, World! " << value << " : time = " << end-start<< std::endl;

    start = omp_get_wtime();
    value = altIntegrate(LEFT, RIGHT, SEGMENTS, STEP);
    end = omp_get_wtime();

    std::cout << "Hello, World! " << value << " : time = " << end-start<< std::endl;
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

    #pragma omp parallel for reduction(+: evenSegments, oddSegments) num_threads(8)
    for (int i = 1; i < segments; i++ ) {
        i%2 == 0 ? evenSegments += formula(left + step * i) : oddSegments += formula(left + step * i);
    }
    evenSegments *= 2;
    oddSegments *= 4;

    return (step / 3) * (formula(left) + formula(right) + evenSegments + oddSegments);
}