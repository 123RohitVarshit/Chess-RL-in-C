#include "autograd.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

matrix* matrix_create(u32 rows,u32 cols){
    matrix* m = (matrix*)malloc(sizeof(matrix));
    m->rows=rows;
    m->cols=cols;
    m->data=(f32*)calloc(rows*cols,sizeof(f32));
    return m;
}

void free_matrix(matrix* m){
    if(m){
        if(m->data) free(m->data);
        free(m);
    }
}


Var* var_create_leaf_node(u32 rows,u32 cols,b32 requires_grad){
    Var* v =(Var*)malloc(sizeof(Var));
    v->val=matrix_create(rows,cols);
    v->grad=requires_grad?matrix_create(rows,cols):NULL;
    v->num_inputs=0;
    v->forward=NULL;
    v->backward=NULL;
    v->flags=requires_grad?1:0;
    return v;
}

Var* var_create_op(ForwardFunc fwd,BackwardFunc bwd, Var** inputs,u32 num_inputs,u32 out_rows,u32 out_cols){
    Var* v=(Var*)malloc(sizeof(Var));
    v->val=matrix_create(out_rows,out_cols);
    v->grad=matrix_create(out_rows,out_cols);
    v->num_inputs=num_inputs;
    for (u32 i=0;i<num_inputs;i++){
        v->inputs[i]=inputs[i];
    }
    v->forward=fwd;
    v->backward=bwd;
    return v;
}

void add_forward(Var* self){
    Var* a=self->inputs[0];
    Var* b=self->inputs[1];
    u32 size=a->val->rows*a->val->cols;
    for (u32 i=0;i<size;i++){
        self->val->data[i]=a->val->data[i]+b->val->data[i];
    }
}

void add_backward(Var* self){
    Var* a=self->inputs[0];
    Var* b=self->inputs[1];
    u32 size=a->val->rows*a->val->cols;
    for (u32 i=0;i<size;i++){
        if(a->grad) a->grad->data[i]+=self->grad->data[i];
        if(b->grad) b->grad->data[i]+=self->grad->data[i];
    }
}

void matmul_forward(Var* self){
    Var* a=self->inputs[0];
    Var* b=self->inputs[1];
    for (u32 i=0;i<a->val->rows;i++){
        for (u32 j=0;j<b->val->cols;j++){
            f32 sum=0.0f;
            for (u32 k=0;k<a->val->cols;k++){
                f32 a_val=a->val->data[i*a->val->cols+k];
                f32 b_val=b->val->data[k*b->val->cols+j];
                sum+=a_val*b_val;
            }
            self->val->data[i*b->val->cols+j]=sum;
        }
    }
}

void matmul_backward(Var* self){
    Var* a=self->inputs[0];
    Var* b=self->inputs[1];

    if(a->grad){
        for (u32 i=0;i<a->val->rows;i++){
            for (u32 k=0;k<a->val->cols;k++){
                f32 sum=0.0f;
                for (u32 j=0;j<b->val->cols;j++){
                    sum+=self->grad->data[i*b->val->cols+j]*b->val->data[k*b->val->cols+j];
                }
                a->grad->data[i*a->val->cols+k]+=sum;
            }
        }
    }
    if (b->grad){
        for (u32 k=0;k<b->val->rows;k++){
            for(u32 j=0;j<b->val->cols;j++){
                f32 sum=0.0f;
                for (u32 i=0;i<a->val->rows;i++){
                    sum+=a->val->data[i*a->val->cols+k]*self->grad->data[i*b->val->cols+j];
                }
                b->grad->data[k*b->val->cols+j]+=sum;
            }
        }
    }
}

void build_topo(Var* v, Var** topo, int* topo_index);

void build_topo(Var* v, Var** topo, int* topo_index) {
    // Check if Bit 1 (value 2) is set. If so, we already visited this node.
    if (v->flags & 2) return; 
    
    // Mark as visited by setting Bit 1
    v->flags |= 2; 

    for (u32 i = 0; i < v->num_inputs; i++) {
        if (v->inputs[i]) {
            build_topo(v->inputs[i], topo, topo_index);
        }
    }
    topo[*topo_index] = v;
    (*topo_index)++;
}

void var_backward(Var* loss) {
    #define MAX_NODES 10000 // Increased just in case, but we don't rely on it for indexing anymore
    Var* topo[MAX_NODES];
    int topo_index = 0;

    build_topo(loss, topo, &topo_index);

    // ONLY initialize gradient if it doesn't exist. 
    // This respects the manual gradient you set in rl.c!
    if (!loss->grad) {
        loss->grad = matrix_create(loss->val->rows, loss->val->cols);
        u32 loss_size = loss->val->rows * loss->val->cols;
        for (u32 i = 0; i < loss_size; i++) {
            loss->grad->data[i] = 1.0f;
        }
    }

    for (int i = topo_index - 1; i >= 0; i--) {
        Var* node = topo[i];
        if (node->backward) {
            node->backward(node);
        }
        // Clear the visited bit (Bit 1) so the node can be used in the next game
        node->flags &= ~2; 
    }
}

void relu_forward(Var* self){
    Var* a=self->inputs[0];
    u32 size=a->val->rows*a->val->cols;
    for (u32 i=0;i<size;i++){
        self->val->data[i]=a->val->data[i]>0.0f?a->val->data[i]:0.0f;
    }
}

void relu_backward(Var* self){
    Var* a=self->inputs[0];
    u32 size=a->val->rows*a->val->cols;
    for (u32 i=0;i<size;i++){
        if(a->grad && a->val->data[i]>0.0f){
            a->grad->data[i]+=self->grad->data[i];
        }
    }
}

void sgd_step(Var** params, u32 num_params, f32 learning_rate) {
    for (u32 p = 0; p < num_params; p++) {
        Var* param = params[p];
        if (!param->grad) continue; // Skip if no gradient needed
        
        u32 size = param->val->rows * param->val->cols;
        for (u32 i = 0; i < size; i++) {
            // 1. Update the weight: weight = weight - (lr * gradient)
            param->val->data[i] -= learning_rate * param->grad->data[i];
            
            // 2. Zero out the gradient for the next training step
            param->grad->data[i] = 0.0f;
        }
    }
}

void free_var(Var* v) {
    if (!v) return;
    free_matrix(v->val);
    free_matrix(v->grad);
    free(v);
}

Var* relu(Var* a){
    Var* inputs[1]={a};
    return var_create_op(relu_forward,relu_backward,inputs,1,a->val->rows,a->val->cols);
}

Var* matmul(Var* a,Var* b){
    Var* inputs[2]={a,b};
    return var_create_op(matmul_forward,matmul_backward,inputs,2,a->val->rows,b->val->cols);
}

Var* add(Var* a,Var* b){
    Var* inputs[2]={a,b};
    return var_create_op(add_forward,add_backward,inputs,2,a->val->rows,a->val->cols);
}
