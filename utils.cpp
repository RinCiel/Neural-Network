#include "utils.h"

// get largest value in the vector
double getMax(std::vector<double> vec) {
	double res = vec[0];
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] > res) {
			res = vec[i];
		}
	}
	return res;
}

int getArgMax(std::vector<double> vec) {
	int index = 0;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] > vec[index]) {
			index = i;
		}
	}
	return index;
}

// get the sum of the elements in the vector
double getSum(std::vector<double> vec) {
	double res = 0;
	for (int i = 0; i < vec.size(); i++) {
		res += vec[i];
	}
	return res;
}

// multiply a 2d vector by a double
std::vector<std::vector<double>> multiply(std::vector<std::vector<double>> vec, double val) {
	std::vector<std::vector<double>> res;
	for (int i = 0; i < vec.size(); i++) {
		std::vector<double> row;
		for (int j = 0; j < vec[i].size(); j++) {
			row.push_back(vec[i][j] * val);
		}
		res.push_back(row);
	}
	return res;
}

// multiply 2 2d vectors
std::vector<std::vector<double>> multiply(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2) {
	std::vector<std::vector<double>> res;
	std::vector<double> row;
	for (int i = 0; i < v1.size(); i++) {
		for (int j = 0; j < v1[i].size(); j++) {
			row.push_back(v1[i][j] * v2[i][j]);
		}
		res.push_back(row);
		row.clear();
	}
	return res;
}

// divide a 2d vector by a double
std::vector<std::vector<double>> divide(std::vector<std::vector<double>> vec, double val) {
	std::vector<std::vector<double>> res;
	for (int i = 0; i < vec.size(); i++) {
		std::vector<double> row;
		for (int j = 0; j < vec[i].size(); j++) {
			row.push_back(vec[i][j] / val);
		}
		res.push_back(row);
	}
	return res;
}

// divide 2 2d vectors
std::vector<std::vector<double>> divide(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2) {
	std::vector<std::vector<double>> res;
	std::vector<double> row;
	for (int i = 0; i < v1.size(); i++) {
		for (int j = 0; j < v1[i].size(); j++) {
			row.push_back(v1[i][j] / v2[i][j]);
		}
		res.push_back(row);
		row.clear();
	}
	return res;
}

// set values to a certain size
std::vector<double> clip(double min, double max, std::vector<double> vec) {
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] < min) {
			vec[i] = min;
		}
		else if (vec[i] > max) {
			vec[i] = max;
		}
	}
	return vec;
}

// dot product of 2 2d vectors
std::vector<std::vector<double>> dot(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2) {
    std::vector<std::vector<double>> res;
    std::vector<double> row;
    for (int i = 0; i < v1.size(); i++) {
        for (int j = 0; j < v2[0].size(); j++) {
            double sum = 0;
            for (int k = 0; k < v1[i].size(); k++) {
                sum += v1[i][k] * v2[k][j];
            }
            row.push_back(sum);
        }
        res.push_back(row);
        row.clear();
    }
    return res;
}

// transpose a 2d vector
std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> vec) {
    std::vector<std::vector<double>> res;
    std::vector<double> row;
    for (int i = 0; i < vec[0].size(); i++) {
        for (int j = 0; j < vec.size(); j++) {
            row.push_back(vec[j][i]);
        }
        res.push_back(row);
        row.clear();
    }
    return res;
}

std::vector<std::vector<double>> transpose(std::vector<double> vec) {
    std::vector<std::vector<double>> res;
    std::vector<double> row;
    for (int i = 0; i < vec.size(); i++) {
        row.push_back(vec[i]);
    }
    res.push_back(row);
    return res;
}
