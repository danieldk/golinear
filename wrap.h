// Copyright 2015 The golinear Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

#ifndef WRAP_H
#define WRAP_H

#include <linear.h>
#include <stddef.h>

typedef struct feature_node feature_node_t;
typedef struct problem problem_t;
typedef struct parameter parameter_t;
typedef struct model model_t;

feature_node_t *nodes_new(size_t n);
void nodes_free(feature_node_t *nodes);
void nodes_put(feature_node_t *nodes, size_t node_idx, int idx,
  double value);
feature_node_t nodes_get(feature_node_t *nodes, size_t idx);

feature_node_t *nodes_vector_get(problem_t *problem, size_t idx);

problem_t *problem_new();
void problem_free(problem_t *problem);
void problem_add_train_inst(problem_t *problem, feature_node_t *nodes,
  double label);

double problem_bias(problem_t *problem);
void set_problem_bias(problem_t *problem, double bias);

parameter_t *parameter_new();
void parameter_set_nthreads(parameter_t *param, int nthreads);
void parameter_free(parameter_t *param);

int *labels_new(int n);

double *probs_new(model_t *model);
double *double_new(size_t n);

double get_double_idx(double *arr, int idx);
int get_int_idx(int *arr, int idx);

void set_double_idx(double *arr, int idx, double val);
void set_int_idx(int *arr, int idx, int val);

char const *check_parameter_wrap(problem_t *prob,
    parameter_t *param);
void cross_validation_wrap(problem_t const *prob, parameter_t const *param,
	int nr_fold, double *target);
void destroy_param_wrap(parameter_t* param);
void get_labels_wrap(model_t const *model, int *label);
int get_nr_class_wrap(model_t const *model);
model_t *load_model_wrap(char const *filename);
double predict_wrap(model_t const *model, feature_node_t *nodes);
double predict_probability_wrap(model_t const *model,
	feature_node_t const *x, double *prob_estimates);
double predict_values_wrap(model_t const *model,
    feature_node_t const *x, double *dec_values);
int save_model_wrap(model_t const *model, char const *filename);
model_t *train_wrap(problem_t *prob, parameter_t *param);
void free_and_destroy_model_wrap(model_t *model);

#endif // WRAP_H
