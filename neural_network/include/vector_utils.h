#pragma once
#include <iostream>
#include <vector>
#include <cmath>

double getMax(std::vector<double> vec);
int getArgMax(std::vector<double> vec);
double getSum(std::vector<double> vec);

std::vector<double> add(std::vector<double> vec, double num);
std::vector<std::vector<double>> add(std::vector<std::vector<double>> vec, double num);
std::vector<std::vector<double>> add(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2);
std::vector<double> add(std::vector<double> v1, std::vector<double> v2);

std::vector<std::vector<double>> subtract(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2);

std::vector<double> multiply(std::vector<double> vec, double val);
std::vector<std::vector<double>> multiply(std::vector<std::vector<double>> vec, double val);
std::vector<std::vector<double>> multiply(std::vector<std::vector<double>> vec, std::vector<double> val);
std::vector<double> multiply(std::vector<double> v1, std::vector<double> v2);
std::vector<std::vector<double>> multiply(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2);

std::vector<double> divide(std::vector<double> v1, std::vector<double> v2);
std::vector<double> divide(std::vector<double> vec, double val);
std::vector<std::vector<double>> divide(std::vector<std::vector<double>> vec, double val);
std::vector<std::vector<double>> divide(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2);

std::vector<double> clip(double min, double max, std::vector<double> vec);

std::vector<double> dot(std::vector<std::vector<double>> vec, std::vector<double> val);
std::vector<std::vector<double>> dot(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2);

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> vec);
std::vector<std::vector<double>> transpose(std::vector<double> vec);

std::vector<double> sumVertical(std::vector<std::vector<double>> vec);

std::vector<std::vector<double>> eye(int n);

std::vector<std::vector<double>> diagflat(std::vector<double> vec);
std::vector<std::vector<double>> diagflat(std::vector<std::vector<double>> vec);

std::vector<double> sqrt(std::vector<double> vec);
std::vector<std::vector<double>> sqrt(std::vector<std::vector<double>> vec);

std::vector<double> sign(std::vector<double> vec);
std::vector<std::vector<double>> sign(std::vector<std::vector<double>> vec);

void print(std::vector<double> vec);
void print(std::vector<std::vector<double>> vec);
