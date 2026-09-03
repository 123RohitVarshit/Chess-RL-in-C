// autograd.h
#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <stdint.h>
#include <stdbool.h>

typedef uint32_t u32;
typedef float    f32;
typedef bool     b32;

// --- Matrix Struct & Functions ---
typedef struct matrix {
    u32 rows;
    u32 cols;
    f32* data; 
} matrix;

matrix* matrix_create(u32 rows, u32 cols);
void free_matrix(matrix* m);

// --- Var Struct & Functions ---
typedef struct Var Var;
typedef void(*ForwardFunc)(Var* self);
typedef void(*BackwardFunc)(Var* self);

struct Var {
    u32 index;
    u32 flags;
    matrix* val;
    matrix* grad;
    Var* inputs[4];
    u32 num_inputs;
    ForwardFunc forward;
    BackwardFunc backward;
};

Var* var_create_leaf_node(u32 rows, u32 cols, b32 requires_grad);
Var* var_create_op(ForwardFunc fwd, BackwardFunc bwd, Var** inputs, u32 num_inputs, u32 out_rows, u32 out_cols);
void free_var(Var* v);

// --- Operations ---
Var* add(Var* a, Var* b);
Var* matmul(Var* a, Var* b);
Var* relu(Var* a);

// --- Engine ---
void var_backward(Var* loss);
void sgd_step(Var** params, u32 num_params, f32 learning_rate);

#endif