#include "..\include\vector_utils.h"

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

// add a 1d vector and a double
std::vector<double> add(std::vector<double> vec, double num) {
	std::vector<double> res(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		res[i] = vec[i] + num;
	}
	return res;
}

// add a 2d vector and a double
std::vector<std::vector<double>> add(std::vector<std::vector<double>> vec, double num) {
	std::vector<std::vector<double>> res(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		res[i] = add(vec[i], num);
	}
	return res;
}

// add 2 1d vectors
std::vector<double> add(std::vector<double> v1, std::vector<double> v2) {
	std::vector<double> res(v1.size());
	for (int i = 0; i < v1.size(); i++) {
		res[i] = v1[i] + v2[i];
	}
	return res;
}

// add 2 2d vectors
std::vector<std::vector<double>> add(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2) {
	std::vector<std::vector<double>> res(v1.size(), std::vector<double>(v1[0].size(), 0));
	for (int i = 0; i < v1.size(); i++) {
		for (int j = 0; j < v1[0].size(); j++) {
			res[i][j] = v1[i][j] + v2[i][j];
		}
	}
	return res;
}

// subtract 2 2d vectors
std::vector<std::vector<double>> subtract(std::vector<std::vector<double>> v1, std::vector<std::vector<double>> v2) {
	std::vector<std::vector<double>> res(v1.size(), std::vector<double>(v1[0].size(), 0));
	for (int i = 0; i < v1.size(); i++) {
		for (int j = 0; j < v1[0].size(); j++) {
			res[i][j] = v1[i][j] - v2[i][j];
		}
	}
	return res;
}

// get the sum of the elements in the vector
double getSum(std::vector<double> vec) {
	double res = 0;
	for (int i = 0; i < vec.size(); i++) {
		res += vec[i];
	}
	return res;
}

// multiply a 1d vector by a double
std::vector<double> multiply(std::vector<double> vec, double val) {
	std::vector<double> res(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		res[i] = vec[i] * val;
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

// multiply a 2d vector by a 1d vector
std::vector<std::vector<double>> multiply(std::vector<std::vector<double>> vec, std::vector<double> val) {
	std::vector<std::vector<double>> res;
	for (int i = 0; i < vec.size(); i++) {
		std::vector<double> row;
		for (int j = 0; j < vec[i].size(); j++) {
			row.push_back(vec[i][j] * val[j]);
		}
		res.push_back(row);
	}
	return res;
}

// multiple 2 1d vectors
std::vector<double> multiply(std::vector<double> v1, std::vector<double> v2) {
	std::vector<double> res(v1.size());
	for (int i = 0; i < v1.size(); i++) {
		res[i] = v1[i] * v2[i];
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

// divide 1d vector by 1d vector
std::vector<double> divide(std::vector<double> v1, std::vector<double> v2) {
	std::vector<double> res(v1.size());
	for (int i = 0; i < v1.size(); i++) {
		res[i] = v1[i] / v2[i];
	}
	return res;
}

// divide a 1d vector by a double
std::vector<double> divide(std::vector<double> vec, double val) {
	std::vector<double> res(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		res[i] = vec[i] / val;
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

// dot product of a 2d vector and a 1d vector, returning a 1d vector
std::vector<double> dot(std::vector<std::vector<double>> vec, std::vector<double> val) {
	std::vector<double> res;
	for (int i = 0; i < vec.size(); i++) {
		double sum = 0;
		for (int j = 0; j < vec[i].size(); j++) {
			sum += vec[i][j] * val[j];
		}
		res.push_back(sum);
	}
	return res;
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
		res.push_back(row);
		row.clear();
    }
    return res;
}

// return a vector as the sum of a 2d vector calculated verticaly
std::vector<double> sumVertical(std::vector<std::vector<double>> vec) {
	std::vector<double> res;
	for (int i = 0; i < vec[0].size(); i++) {
		double sum = 0;
		for (int j = 0; j < vec.size(); j++) {
			sum += vec[j][i];
		}
		res.push_back(sum);
	}
	return res;
}

// numpy's eye
std::vector<std::vector<double>> eye(int n) {
	std::vector<std::vector<double>> res;
	std::vector<double> row;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) {
				row.push_back(1);
			}
			else {
				row.push_back(0);
			}
		}
		res.push_back(row);
		row.clear();
	}
	return res;
}

// numpy's diagflat
std::vector<std::vector<double>> diagflat(std::vector<double> vec) {
	std::vector<std::vector<double>> res = eye(vec.size());
	return multiply(res, vec);
}

std::vector<std::vector<double>> diagflat(std::vector<std::vector<double>> vec) {
	// change vec to a 1d vector
	std::vector<double> vec1d;
	for (int i = 0; i < vec.size(); i++) {
		for (int j = 0; j < vec[i].size(); j++) {
			vec1d.push_back(vec[i][j]);
		}
	}
	return diagflat(vec1d);
}

// square root a 1d vector
std::vector<double> sqrt(std::vector<double> vec) {
	std::vector<double> res;
	for (int i = 0; i < vec.size(); i++) {
		res.push_back(sqrt(vec[i]));
	}
	return res;
}

// square root a 2d vector
std::vector<std::vector<double>> sqrt(std::vector<std::vector<double>> vec) {
	std::vector<std::vector<double>> res;
	std::vector<double> row;
	for (int i = 0; i < vec.size(); i++) {
		for (int j = 0; j < vec[i].size(); j++) {
			row.push_back(sqrt(vec[i][j]));
		}
		res.push_back(row);
		row.clear();
	}
	return res;
}

// print out 1d vectors
void print(std::vector<double> vec) {
	for (int i = 0; i < vec.size(); i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << std::endl;
}

// print out 2d vectors
void print(std::vector<std::vector<double>> vec) {
	for (int i = 0; i < vec.size(); i++) {
		for (int j = 0; j < vec[i].size(); j++) {
			std::cout << vec[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

// if element in 1d vector >=0 , set to 1, else set to -1
std::vector<double> sign(std::vector<double> vec) {
	std::vector<double> res;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] >= 0) {
			res.push_back(1);
		}
		else {
			res.push_back(-1);
		}
	}
	return res;
}

// if element in 2d vector >=0 , set to 1, else set to -1
std::vector<std::vector<double>> sign(std::vector<std::vector<double>> vec) {
	std::vector<std::vector<double>> res;
	std::vector<double> row;
	for (int i = 0; i < vec.size(); i++) {
		for (int j = 0; j < vec[i].size(); j++) {
			if (vec[i][j] >= 0) {
				row.push_back(1);
			}
			else {
				row.push_back(-1);
			}
		}
		res.push_back(row);
		row.clear();
	}
	return res;
}