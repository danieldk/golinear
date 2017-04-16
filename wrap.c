// Copyright 2015 The golinear Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

#include <stdlib.h>
#include <string.h>

#include <linear.h>

#include "wrap.h"

#ifdef CV_OMP
#include <omp.h>
#endif

feature_node_t *nodes_new(size_t n)
{
  feature_node_t *nodes = malloc((n + 1) * sizeof(feature_node_t));
  if (nodes == NULL) {
    return NULL;
  }

  // Terminator
  nodes[n].index = -1;
  nodes[n].value = 0.0;

  return nodes;
}

void nodes_free(feature_node_t *nodes)
{
  free(nodes);
}

void nodes_put(feature_node_t *nodes, size_t nodes_idx, int idx,
  double value)
{
  nodes[nodes_idx].index = idx;
  nodes[nodes_idx].value = value;
}

feature_node_t nodes_get(feature_node_t *nodes, size_t idx)
{
  return nodes[idx];
}

feature_node_t *nodes_vector_get(problem_t *problem, size_t idx)
{
  return problem->x[idx];
}

problem_t *problem_new()
{
  problem_t *problem = malloc(sizeof(problem_t));
  if (problem == NULL) {
    return NULL;
  }

  problem->l = 0;
  problem->n = 0;
  problem->bias = -1;
  problem->y = malloc(0);
  if (problem->y == NULL) {
    free(problem);
    return NULL;
  }
  problem->x = malloc(0);
  if (problem->x == NULL) {
    free(problem->y);
    free(problem);
    return NULL;
  }

  return problem;
}

void problem_free(problem_t *problem)
{
  free(problem->x);
  free(problem->y);
  free(problem);
}

void problem_add_train_inst(problem_t *problem, feature_node_t *nodes,
  double label)
{
  ++problem->l;

  // The number of features equals the highest feature index.
  feature_node_t *node;
  for (node = nodes; node->index != -1; ++node)
  	if (node->index > problem->n)
  		problem->n = node->index;

  problem->y = realloc(problem->y, (size_t) problem->l * sizeof(double));
  problem->y[problem->l - 1] = label;
  problem->x = realloc(problem->x, (size_t) problem->l * sizeof(feature_node_t *));
  problem->x[problem->l - 1] = nodes;
}

double problem_bias(problem_t *problem)
{
  return problem->bias;
}

void set_problem_bias(problem_t *problem, double bias)
{
  problem->bias = bias;
}

parameter_t *parameter_new()
{
  parameter_t *param = malloc(sizeof(parameter_t));
  if (param == NULL) {
    return NULL;
  }
  memset(param, 0, sizeof(parameter_t));
  return param;
}

void parameter_set_nthreads(parameter_t *param, int nthreads)
{
#ifdef CV_OMP
  if (nthreads <= 0)
    param->nr_thread = omp_get_max_threads();
  else
    param->nr_thread = nthreads;
#endif
}

void parameter_free(parameter_t *param)
{
  if (param->weight_label != NULL) {
    free(param->weight_label);
    param->weight_label = NULL;
  }
  if (param->weight != NULL) {
    free(param->weight);
    param->weight = NULL;
  }
}

double *double_new(size_t n)
{
  double *r = malloc(n * sizeof(double));
  if (r == NULL) {
    return NULL;
  }
  memset(r, 0, n * sizeof(double));
  return r;
}

int *labels_new(int n)
{
  int *labels = malloc((size_t) n * sizeof(int));
  if (labels == NULL) {
    return NULL;
  }
  memset(labels, 0, (size_t) n * sizeof(int));
  return labels;
}

double *probs_new(model_t *model)
{
  int nClasses = get_nr_class(model);
  double *probs = malloc((size_t) nClasses * sizeof(double));
  if (probs == NULL) {
      return NULL;
  }
  memset(probs, 0, (size_t) nClasses * sizeof(double));
  return probs;
}

double get_double_idx(double *arr, int idx)
{
  return arr[idx];
}

int get_int_idx(int *arr, int idx)
{
  return arr[idx];
}

void set_double_idx(double *arr, int idx, double val)
{
  arr[idx] = val;
}

void set_int_idx(int *arr, int idx, int val)
{
  arr[idx] = val;
}

char const *check_parameter_wrap(problem_t *prob, parameter_t *param)
{
  return check_parameter(prob, param);
}

void cross_validation_wrap(problem_t const *prob, parameter_t const *param,
  int nr_fold, double *target)
{
  return cross_validation(prob, param, nr_fold, target);
}


void destroy_param_wrap(parameter_t* param)
{
  return destroy_param(param);
}

void free_and_destroy_model_wrap(model_t *model)
{
  free_and_destroy_model(&model);
}

void get_labels_wrap(model_t const *model, int *label)
{
  get_labels(model, label);
}

int get_nr_class_wrap(model_t const *model)
{
  return get_nr_class(model);
}

model_t *load_model_wrap(char const *filename)
{
  return load_model(filename);
}

double predict_probability_wrap(model_t const *model, 
    feature_node_t const *x, double *prob_estimates)
{
  return predict_probability(model, x, prob_estimates);
}

double predict_values_wrap(model_t const *model,
    feature_node_t const *x, double *dec_values)
{
  return predict_values(model, x, dec_values);
}

int save_model_wrap(model_t const *model, char const *filename)
{
  return save_model(filename, model);
}

model_t *train_wrap(problem_t *prob, parameter_t *param)
{
  return train(prob, param);
}

double predict_wrap(model_t const *model, feature_node_t *nodes)
{
  return predict(model, nodes);
}
