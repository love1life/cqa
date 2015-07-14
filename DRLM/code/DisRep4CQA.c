//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined _WIN32
# include "win32-port.h"
#else
# include <pthread.h>
#endif

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 300
#define MAX_ANSWER_COUNT 20
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int category_hash_size = 10000;

typedef float real;                    // Precision of float numbers

struct vocab_word {
	long long cn;
	int outsignal;
	int *point;
	char *word, *code, codelen;
};

struct category_word
{
	char *word;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int *vocab_hash;

struct  category_word *category;
int *category_hash;

int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1, eproch = 1;

long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100, category_max_size = 1000, category_size = 0;
long long train_words_actual = 0, train_words = 0, word_count_actual = 0, file_size = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0c, *syn0v, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
	int a, i;
	long long train_words_pow = 0;
	real d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));
	for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
	i = 0;
	d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (real)table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
		}
		if (i >= vocab_size) i = vocab_size - 1;
	}
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries, 
// tab is the source and target pairs boundary,
// EOL is the sentence boundary
int ReadWord(char *word, FILE *fin)
{
	int a = 0, ch = 0;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) break;
			else continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	word[a] = '\0';
	return ch;
}

// Returns hash value of a word
int GetWordHash(char *word, int hash_size) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word, vocab_hash_size);
	while (1) {
		if (vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

int SearchCategory(char *word)
{
	unsigned int hash = GetWordHash(word, category_hash_size);
	while (1) {
		if (category_hash[hash] == -1) return -1;
		if (!strcmp(word, category[category_hash[hash]].word)) return category_hash[hash];
		hash = (hash + 1) % category_hash_size;
	}
	return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word);
}

int ReadSentence(int *category, int *que, int ans[][MAX_SENTENCE_LENGTH + 1], FILE *fin)
{
	char word[MAX_STRING];
	int a = 0, b = 0, c = 0, id = -1, ch;
	ReadWord(word, fin);
	if (feof(fin)) return 0;
	c = SearchCategory(word);
	*category = c;
	//read question
	c = 0;
	while (1)
	{
		ch = ReadWord(word, fin);
		if (feof(fin)) return 0;
		a++;
		id = SearchVocab(word);
		if (id != -1 && c < MAX_SENTENCE_LENGTH) {
			que[c] = id;
			c++;
		}
		if (ch == '\t' || ch == '\n')
			break;
	}
	que[c] = -1;
	if (c == 0) *category = -1;
	//read answers
	b = 0;
	while (1)
	{
		c = 0;
		while (1)
		{
			ch = ReadWord(word, fin);
			if (feof(fin)) return 0;
			a++;
			id = SearchVocab(word);
			if (id != -1 && c < MAX_SENTENCE_LENGTH) {
				ans[b][c] = id;
				//printf("%d\r", id);
				c++;
			}
			if (ch == '\t' || ch == '\n')
				break;
		}
		ans[b][c] = -1;
		if (c != 0 && b < MAX_ANSWER_COUNT)
			b++;
		if (ch == '\n')
			break;
	}
	ans[b][0] = -1;
	return a;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;
	vocab[vocab_size].outsignal = 0;
	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word, vocab_hash_size);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}

int AddWordToCategory(char *word)
{
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	category[category_size].word = (char *)calloc(length, sizeof(char));
	strcpy(category[category_size].word, word);
	category_size++;
	// Reallocate memory if needed
	if (category_size + 2 >= category_max_size) {
		category_max_size += 1000;
		category = (struct vocab_word *)realloc(category, category_max_size * sizeof(struct category_word));
	}
	hash = GetWordHash(word, category_hash_size);
	while (category_hash[hash] != -1) hash = (hash + 1) % category_hash_size;
	category_hash[hash] = category_size - 1;
	return category_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	size = vocab_size;
	train_words_actual = 0;
	train_words = 0;
	for (a = 1; a < size; a++) {
		train_words += vocab[a].cn;
		// Words occurring less than min_count times will be discarded from the vocab
		if (vocab[a].cn < min_count) {
			vocab_size--;
			free(vocab[a].word);
			vocab[a].word = NULL;
		}
		else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash = GetWordHash(vocab[a].word, vocab_hash_size);
			while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = a;
			train_words_actual += vocab[a].cn;
		}
	}
	vocab = (struct vocab_word *)realloc(vocab, vocab_size * sizeof(struct vocab_word));
	// Allocate memory for the binary tree construction
	for (a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
		vocab[b].cn = vocab[a].cn;
		vocab[b].word = vocab[a].word;
		b++;
	}
	else free(vocab[a].word);
	vocab_size = b;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word, vocab_hash_size);
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
	for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for (a = 0; a < vocab_size - 1; a++) {
		// First, find two smallest nodes 'min1, min2'
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			}
			else {
				min1i = pos2;
				pos2++;
			}
		}
		else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			}
			else {
				min2i = pos2;
				pos2++;
			}
		}
		else {
			min2i = pos2;
			pos2++;
		}
		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b == vocab_size * 2 - 2) break;
		}
		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;
		for (b = 0; b < i; b++) {
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}

void LearnVocabAndCateFromTrainFile() {
	char word[MAX_STRING], ch;
	FILE *fin;
	long long a, i;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < category_hash_size; a++)category_hash[a] = -1;
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size = 0;
	category_size = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		i = SearchCategory(word);
		if (i == -1)
			AddWordToCategory(word);
		while (1)
		{
			ch = ReadWord(word, fin);
			if (feof(fin))
				break;
			train_words_actual++;
			train_words++;
			if ((debug_mode > 1) && (train_words_actual % 100000 == 0)) {
				printf("%lldK%c", train_words_actual / 1000, 13);
				fflush(stdout);
			}
			i = SearchVocab(word);
			if (i == -1) {
				a = AddWordToVocab(word);
				vocab[a].cn = 1;
			}
			else vocab[i].cn++;
			if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
			if (ch == '\n')
				break;
		}
		if (feof(fin))
			break;
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words_actual);
	}
	file_size = ftell(fin);
	fclose(fin);
}

void SaveVocab() {
	long long i;
	FILE *fo = fopen(save_vocab_file, "wb");
	fprintf(fo, "%lld\n", category_size);
	for (i = 0; i < category_size; i++)fprintf(fo, "%s\n", category[i].word);
	for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

void InitNet() {
	long long a, b;
	a = posix_memalign((void **)&syn0v, 128, (long long)vocab_size * layer1_size * sizeof(real));
	if (syn0v == NULL) { printf("Memory allocation failed\n"); exit(1); }
	a = posix_memalign((void **)&syn0c, 128, (long long)category_size * layer1_size * sizeof(real));
	if (syn0c == NULL) { printf("Memory allocation failed\n"); exit(1); }
	if (hs) {
		a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (syn1 == NULL) { printf("Memory allocation failed\n"); exit(1); }
		for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
			syn1[a * layer1_size + b] = 0;
	}
	if (negative > 0) {
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (syn1neg == NULL) { printf("Memory allocation failed\n"); exit(1); }
		for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
			syn1neg[a * layer1_size + b] = 0;
	}
	for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
		syn0v[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
	for (b = 0; b < layer1_size; b++) for (a = 0; a < category_size; a++)
		syn0c[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
	CreateBinaryTree();
}

void *TrainModelThread(void *id) {
	int a, b, d, cate, word, last_word, question_length = 0, question_position = 0, answer_length = 0, answer_position = 0, answer_index = 0;
	int question[MAX_SENTENCE_LENGTH + 1], answers[MAX_ANSWER_COUNT + 1][MAX_SENTENCE_LENGTH + 1];
	long long word_count = 0, last_word_count = 0;
	int *cur_ans = NULL;
	long long cl, l1, l2, c, target, label;
	unsigned long long next_random = (long long)id;
	char cword[MAX_STRING], ch;
	real f, g;
	clock_t now;
	real *neu1 = (real *)calloc(layer1_size, sizeof(real));
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));
	FILE *fi = fopen(train_file, "rb");
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	if ((long long)id != 0) {
		while (1) {
			ch = ReadWord(cword, fi);
			if (feof(fi)) return;
			if (ch == '\n')break;
		}
	}
	while (1) {
		if (word_count - last_word_count > 10000) {
			word_count_actual += word_count - last_word_count;
			last_word_count = word_count;
			if ((debug_mode > 1)) {
				now = clock();
				printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
					word_count_actual / (real)(train_words + 1) * 100,
					word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			alpha = starting_alpha * (1 - word_count_actual / (real)(train_words_actual + 1));
			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
		}
		if (question_length == 0) {
			cate = -1;
			while (cate == -1 && !feof(fi)) word_count += ReadSentence(&cate, question, answers, fi);
			if (!feof(fi))
			{
				for (c = 0; c <= MAX_SENTENCE_LENGTH && question[c] != -1; c++);
				question_length = c;
			}
			question_position = 0;
		}
		if (feof(fi)) break;
		if (word_count > train_words / num_threads) break;
		if (question_length == 0)
			continue;
		word = question[question_position];
		if (word == -1) continue;
		vocab[word].outsignal = 1;
		for (c = 0; c < layer1_size; c++) neu1[c] = syn0c[c + cate * layer1_size];
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
		next_random = next_random * (unsigned long long)25214903917 + 11;
		b = next_random % window;


		//train the question part
		// in -> hidden
		for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
			c = question_position - window + a;
			if (c < 0) continue;
			if (c >= question_length) continue;
			last_word = question[c];
			if (last_word == -1) continue;
			vocab[last_word].outsignal = 1;
			for (c = 0; c < layer1_size; c++) neu1[c] += syn0v[c + last_word * layer1_size];
		}
		if (hs) for (d = 0; d < vocab[word].codelen; d++) {
			f = 0;
			l2 = vocab[word].point[d] * layer1_size;
			// Propagate hidden -> output
			for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
			if (f <= -MAX_EXP) continue;
			else if (f >= MAX_EXP) continue;
			else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
			// 'g' is the gradient multiplied by the learning rate
			g = (1 - vocab[word].code[d] - f) * alpha;
			// Propagate errors output -> hidden
			for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
			// Learn weights hidden -> output
			for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
		}
		// NEGATIVE SAMPLING
		if (negative > 0) for (d = 0; d < negative + 1; d++) {
			if (d == 0) {
				target = word;
				label = 1;
			}
			else {
				next_random = next_random * (unsigned long long)25214903917 + 11;
				target = table[(next_random >> 16) % table_size];
				if (target == 0) target = next_random % (vocab_size - 1) + 1;
				if (target == word) continue;
				label = 0;
			}
			l2 = target * layer1_size;
			f = 0;
			for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
			if (f > MAX_EXP) g = (label - 1) * alpha;
			else if (f < -MAX_EXP) g = (label - 0) * alpha;
			else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
			for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
			for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
		}
		// hidden -> in
		for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
			c = question_position - window + a;
			if (c < 0) continue;
			if (c >= question_length) continue;
			last_word = question[c];
			if (last_word == -1) continue;
			for (c = 0; c < layer1_size; c++) syn0v[c + last_word * layer1_size] += neu1e[c];
		}
		for (c = 0; c < layer1_size; c++) syn0c[c + cate * layer1_size] += neu1e[c];


		//train the answers part
		answer_index = 0;
		answer_position = -1;
		answer_length = 0;
		while (answers[answer_index][0] != -1) {
			if (answer_length == 0)
			{
				cur_ans = (int*)(answers[answer_index]);
				for (c = 0; c <= MAX_SENTENCE_LENGTH&&cur_ans[c] != -1; c++);
				answer_length = c;
			}
			answer_position++;
			while (answer_position < answer_length && cur_ans[answer_position] != word)answer_position++;
			if (answer_position >= answer_length) {
				answer_length = 0;
				answer_position = -1;
				answer_index++;
				continue;
			}
			for (c = 0; c < layer1_size; c++) neu1[c] = syn0c[c + cate * layer1_size];
			for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
			next_random = next_random * (unsigned long long)25214903917 + 11;
			b = next_random % window;
			// in -> hidden
			for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
				c = answer_position - window + a;
				if (c < 0) continue;
				if (c >= answer_length) continue;
				last_word = cur_ans[c];
				if (last_word == -1) continue;
				vocab[last_word].outsignal = 1;
				for (c = 0; c < layer1_size; c++) neu1[c] += syn0v[c + last_word * layer1_size];
			}
			if (hs) for (d = 0; d < vocab[word].codelen; d++) {
				f = 0;
				l2 = vocab[word].point[d] * layer1_size;
				// Propagate hidden -> output
				for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				// 'g' is the gradient multiplied by the learning rate
				g = (1 - vocab[word].code[d] - f) * alpha;
				// Propagate errors output -> hidden
				for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
				// Learn weights hidden -> output
				for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
			}
			// NEGATIVE SAMPLING
			if (negative > 0) for (d = 0; d < negative + 1; d++) {
				if (d == 0) {
					target = word;
					label = 1;
				}
				else {
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];
					if (target == 0) target = next_random % (vocab_size - 1) + 1;
					if (target == word) continue;
					label = 0;
				}
				l2 = target * layer1_size;
				f = 0;
				for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
				if (f > MAX_EXP) g = (label - 1) * alpha;
				else if (f < -MAX_EXP) g = (label - 0) * alpha;
				else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
				for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
				for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
			}
			// hidden -> in
			for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
				c = answer_position - window + a;
				if (c < 0) continue;
				if (c >= answer_length) continue;
				last_word = cur_ans[c];
				if (last_word == -1) continue;
				for (c = 0; c < layer1_size; c++) syn0v[c + last_word * layer1_size] += neu1e[c];
			}
			for (c = 0; c < layer1_size; c++) syn0c[c + cate * layer1_size] += neu1e[c];
		}
		question_position++;
		if (question_position >= question_length) {
			question_length = 0;
			continue;
		}
	}
	fclose(fi);
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
}

void ReadVocab() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING], file[MAX_STRING];
	FILE *fin = NULL;

	sprintf(file, "%s.cate", read_vocab_file);
	fin = fopen(file, "rb");
	if (fin == NULL) {
		printf("Category file not found\n");
		exit(1);
	}

	for (a = 0; a < category_hash_size; a++)category_hash[a] = -1;
	category_size = 0;
	while (1)
	{
		c = ReadWord(word, fin);
		//printf("%s\r", word);
		if (feof(fin)) break;
		AddWordToCategory(word);
		while (c != '\n'&&!feof(fin)){
			fscanf(fin, "%c", &c);
			//printf("%c\r", c);
		}
	}
	if (debug_mode > 0) {
		printf("Category size: %lld\n", category_size);
	}
	fclose(fin);
	sprintf(file, "%s.voc", read_vocab_file);
	fin = fopen(file, "rb");
	if (fin == NULL) {
		printf("Vocaborary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	i = 0;
	vocab_size = 0;
	while (1) {
		c = ReadWord(word, fin);
		//printf("%s\r", word);
		if (feof(fin)) break;
		a = AddWordToVocab(word);
		fscanf(fin, "%lld%c", &vocab[a].cn, &c);
		while (c != '\n'&&!feof(fin)){
			fscanf(fin, "%c", &c);
			//printf("%c\r", c);
		}
		i++;
		if ((debug_mode > 1) && (i % 100000 == 0)) {
			printf("%lldK%c", i / 1000, 13);
			fflush(stdout);
		}
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	fclose(fin);
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);
	fclose(fin);
}

void TrainModel() {
	long a, b, c, d, e;
	char output_final_file[MAX_SENTENCE_LENGTH];
	FILE *fo;

	printf("Starting training using file %s\n", train_file);
	starting_alpha = alpha;
	if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabAndCateFromTrainFile();
	if (save_vocab_file[0] != 0) SaveVocab();
	if (output_file[0] == 0) return;
	InitNet();
	if (negative > 0) InitUnigramTable();
	for (e = 0; e < eproch; e++){
		pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
		start = clock();
		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
		free(pt);
		sprintf(output_final_file, "%s_eproch%d.model", output_file, e);
		fo = fopen(output_final_file, "wb");
		
		c = 0;
		for (a = 0; a < vocab_size; a++) { if (vocab[a].outsignal == 1)c++; }
		// Save the word vectors
		fprintf(fo, "%lld %lld\n", c, layer1_size);
		for (a = 0; a < vocab_size; a++) {
			if (vocab[a].outsignal == 1) {
				fprintf(fo, "%s ", vocab[a].word);
				if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0v[a * layer1_size + b], sizeof(real), 1, fo);
				else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0v[a * layer1_size + b]);
				fprintf(fo, "\n");
			}
		}
		fprintf(fo, "%lld %lld\n", category_size, layer1_size);
		for (a = 0; a < category_size; a++) {
			fprintf(fo, "%s ", category[a].word);
			if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0c[a * layer1_size + b], sizeof(real), 1, fo);
			else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0c[a * layer1_size + b]);
			fprintf(fo, "\n");
		}
		
		fclose(fo);
	}
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-eproch <int>\n");
		printf("\t\tSet eproch of training; default is 1\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
		printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		return 0;
	}
	train_file[0] = 0;
	output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	if ((i = ArgPos((char *)"-eproch", argc, argv)) > 0) eproch = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	eproch = max(1, eproch);
	

	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

	category = (struct category_word*)calloc(category_max_size, sizeof(struct  category_word));
	category_hash = (int*)calloc(category_hash_size, sizeof(int));

	expTable = (real *)malloc(EXP_TABLE_SIZE * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
	TrainModel();
	return 0;
}
