// rl.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "autograd.h"
#include "chess.h"

int main() {
    srand(time(NULL));
    static u32 global_index = 0;

    // 1. Initialize Neural Network Weights (2-layer network)
    Var* W1 = var_create_leaf_node(64, 8, true); W1->index = global_index++;
    Var* W2 = var_create_leaf_node(8, 1, true);  W2->index = global_index++;
    
    for(u32 i = 0; i < 64 * 8; i++) W1->val->data[i] = ((f32)rand() / RAND_MAX - 0.5f) * 0.1f;
    for(u32 i = 0; i < 8 * 1; i++)  W2->val->data[i] = ((f32)rand() / RAND_MAX - 0.5f) * 0.1f;

    Var* params[2] = {W1, W2};
    f32 learning_rate = 0.05f;
    f32 epsilon = 0.3f;

    printf("Starting Micro-Chess RL Training (Modular Autograd)...\n");
    
    int white_wins = 0, black_wins = 0, draws = 0;
    int total_games = 500; 

    for(int game = 0; game < total_games; game++) {
        init_board();
        int current_player = 1; 
        int turn = 0;
        
        // Memory Management: Track every Var node created this game to free them later
        Var* game_nodes[1000]; 
        int node_count = 0;

        while(turn < 100) {
            int moves[256];
            int num_moves = get_legal_moves(current_player, moves);
            if(num_moves == 0) break; 
            
            // --- 1. BUILD GRAPH FOR CURRENT STATE ---
            Var* X = var_create_leaf_node(1, 64, false); X->index = global_index++;
            board_to_tensor(X->val->data);
            game_nodes[node_count++] = X;

            Var* H_raw = matmul(X, W1); H_raw->index = global_index++;
            Var* H = relu(H_raw);       H->index = global_index++;
            Var* Score = matmul(H, W2); Score->index = global_index++;
            
            H_raw->forward(H_raw);
            H->forward(H);
            Score->forward(Score);
            
            game_nodes[node_count++] = H_raw;
            game_nodes[node_count++] = H;
            game_nodes[node_count++] = Score;

            // --- 2. CHOOSE MOVE ---
            int best_move = moves[0];
            f32 best_score = -1e9;
            
            if((f32)rand() / RAND_MAX < epsilon) {
                best_move = moves[rand() % num_moves];
            } else {
                for(int m = 0; m < num_moves; m++) {
                    int from = moves[m] / 64;
                    int to = moves[m] % 64;
                    
                    int captured = make_move(from, to);
                    if(abs(captured) == 2) { best_move = moves[m]; make_move(to, from); break; }
                    
                    // Evaluate temporary move
                    Var* temp_X = var_create_leaf_node(1, 64, false); temp_X->index = global_index++;
                    board_to_tensor(temp_X->val->data);
                    
                    Var* t_H_raw = matmul(temp_X, W1); t_H_raw->index = global_index++;
                    Var* t_H = relu(t_H_raw);          t_H->index = global_index++;
                    Var* t_Score = matmul(t_H, W2);    t_Score->index = global_index++;
                    
                    t_H_raw->forward(t_H_raw); t_H->forward(t_H); t_Score->forward(t_Score);
                    
                    f32 score = t_Score->val->data[0];
                    if(current_player == -1) score = -score; 
                    if(score > best_score) { best_score = score; best_move = moves[m]; }
                    
                    // Free temporary nodes immediately
                    free_var(temp_X); free_var(t_H_raw); free_var(t_H); free_var(t_Score);
                    make_move(to, from); // Undo
                }
            }
            
            // --- 3. EXECUTE MOVE ---
            int from = best_move / 64;
            int to = best_move % 64;
            make_move(from, to);
            
            int winner = check_winner();
            if(winner != 0) {
                if(winner == 1) white_wins++; else black_wins++;
                
                // --- 4. AUTOGRAD TRAINING STEP ---
                f32 target = (winner == 1) ? 1.0f : -1.0f; 
                
                // We only backprop the FINAL score of the game to the network
                // (In advanced RL like PPO, you'd backprop intermediate rewards, but this is fine for now)
                Score->grad->data[0] = 2.0f * (Score->val->data[0] - target); // MSE derivative
                var_backward(Score);
                
                sgd_step(params, 2, learning_rate);
                
                break; // Game over
            }
            
            current_player = -current_player;
            turn++;
        }
        if(turn >= 100) draws++;
        
        // --- 5. CLEANUP: Free all nodes created during this game ---
        for(int i = 0; i < node_count; i++) {
            free_var(game_nodes[i]);
        }
        
        if(epsilon > 0.05f) epsilon -= 0.0005f;

        if(game % 100 == 0) {
            printf("Game %4d | W: %3d | B: %3d | D: %3d | Eps: %.2f\n", game, white_wins, black_wins, draws, epsilon);
        }
    }

    printf("\n--- Training Complete! ---\n");
    printf("Final -> White: %d | Black: %d | Draws: %d\n", white_wins, black_wins, draws);
    
    // Free the actual weights at the very end
    free_var(W1); 
    free_var(W2);
    return 0;
}