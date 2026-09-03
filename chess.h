// chess.h
#ifndef CHESS_H
#define CHESS_H

#define BOARD_SIZE 64

void init_board();
int get_legal_moves(int player, int* moves);
int make_move(int from, int to);
int check_winner();
void board_to_tensor(float* tensor); // Converts board to 1.0, -1.0, 0.0 for the NN

#endif