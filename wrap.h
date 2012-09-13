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

problem_t *problem_new();
void problem_free(problem_t *problem);
void problem_add_train_inst(problem_t *problem, feature_node_t *nodes,
  double label);

parameter_t *parameter_new();

char const *check_parameter_wrap(problem_t *prob,
    parameter_t *param);
void destroy_param_wrap(parameter_t* param);
model_t *load_model_wrap(char const *filename);
double predict_wrap(model_t const *model, feature_node_t *nodes);
int save_model_wrap(model_t const *model, char const *filename);
model_t *train_wrap(problem_t *prob, parameter_t *param);
void free_and_destroy_model_wrap(model_t *model);

#endif // WRAP_H