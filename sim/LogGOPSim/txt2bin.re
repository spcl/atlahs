#include <map>
#include <math.h>
#include <string>
#include <limits>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>

#include "Goal.hpp"
#include "cmdline_txt2bin.h"

typedef unsigned int uint;
typedef unsigned char uchar;

#define	BSIZE	10240

#define	YYCTYPE		uchar
#define	YYCURSOR	cursor
#define	YYLIMIT		s->lim
#define	YYMARKER	s->ptr
#define	YYFILL(n)	{cursor = fill(s, cursor, n);}

#define	RET(i)	{s->cur = cursor; return i;}

gengetopt_args_info args_info;

typedef struct Scanner {
    FILE*		fd;
    uchar		*bot, *tok, *ptr, *cur, *pos, *lim, *top, *eof;
    uint		line;
	int			rank;
	uint32_t 	curr_rank, num_ranks;
	Goal*		schedule;
	std::map<std::string, goalop_t>* idtbl;
} Scanner;

typedef struct Item {
	char type;
	char *label1;
	char *label2;
	uint64_t size;
	uint32_t target;
	uint32_t tag;
	uint8_t cpu;
	uint8_t nic;	
} Item;

enum OpTypes {
	Undefined,
	SendOp,
	RecvOp,
	LoclOp,
	StartDependency,
	Dependency
};

inline uint64_t add_number(unsigned char *s, unsigned char *e) {

	uint64_t num = 0;	
	--s;
	while(++s < e) num = num * 10 + (*s - '0');

	return num;
}

inline void insert_id(Scanner *s, char *id, goalop_t op) {

	s->idtbl->insert(std::make_pair(std::string(id), op));
	free(id);
}

inline goalop_t retrieve_id(Scanner *s, char *id) {

	std::map<std::string, goalop_t>::iterator it;
	it = s->idtbl->find(std::string(id));
	if (it == s->idtbl->end()) {
		fprintf(stderr, "A dependency references label %s, which is undefined!\n", id);
		exit(EXIT_FAILURE);
	}
	free(id);
	return it->second;
} 

void process_item(Scanner *s, Item *item) {

/*
	printf("Parsed Item:\n");
	printf("  Type: ");
	if (item->type == Undefined) printf("Undefined\n");
	if (item->type == SendOp) printf("Send\n");
	if (item->type == RecvOp) printf("Recv\n");
	if (item->type == LoclOp) printf("LoclOp\n");
	if (item->type == StartDependency) printf("IRequires\n");
	if (item->type == Dependency) printf("Requires\n");
	printf("  Label1: %s\n", item->label1);
	printf("  Label2: %s\n", item->label2);
	printf("  Size: %i\n", item->size);
	printf("  Target: %i\n", item->target);
	printf("  Tag: %i\n", item->tag);
	printf("  CPU: %i\n", item->cpu);
	printf("  NIC: %i\n", item->nic);
*/
	
	goalop_t op, op2;

	switch (item->type) {
		case Undefined:
			fprintf(stderr, "Error while parsing, attempt to add a undefined operation\n");
			exit(EXIT_FAILURE);
			break;
		case SendOp:
			op = s->schedule->Send(s->rank, item->target, item->size, item->tag, item->cpu, item->nic);
			if (item->label1 != NULL) insert_id(s, item->label1, op);
			break;
		case RecvOp:
			op = s->schedule->Recv(item->target, s->rank, item->size, item->tag, item->cpu, item->nic);
			if (item->label1 != NULL) insert_id(s, item->label1, op);
			break;
		case LoclOp:
			op = s->schedule->Calc(s->rank, item->size, item->cpu, item->nic);
			if (item->label1 != NULL) insert_id(s, item->label1, op);
			break;
		case StartDependency:
			op = retrieve_id(s, item->label1);
			op2 = retrieve_id(s, item->label2);
			assert(op != NULL);
			assert(op2 != NULL);
			s->schedule->StartDependency(op, op2);
			break;
		case Dependency:
			op = retrieve_id(s, item->label1);
			op2 = retrieve_id(s, item->label2);
			assert(op != NULL);
			assert(op2 != NULL);
			s->schedule->Dependency(op, op2);
			break;
		default:
			break;
	}
}

inline char* add_label(unsigned char *s, unsigned char *e) {
	
	char *buf = NULL;

	buf = (char *) malloc((e-s)+1);
	memcpy((void *) buf, (void *) s, (size_t) (e-s));
	buf[e-s] = '\0';
	
	return buf;

}


uchar *fill(Scanner *s, uchar *cursor, int numtoread) {

	if(!s->eof) {
		uint cnt = s->tok - s->bot;
		if(cnt){
			if ((s->lim - s->tok) < abs(s->bot - s->tok)) memcpy(s->bot, s->tok, s->lim - s->tok);
			else memmove(s->bot, s->tok, s->lim - s->tok);
			s->tok = s->bot;
			s->ptr -= cnt;
			cursor -= cnt;
			s->pos -= cnt;
			s->lim -= cnt;
		}
		if((s->top - s->lim) < BSIZE){
			uchar *buf = (uchar*) malloc(((s->lim - s->bot) + BSIZE)*sizeof(uchar));
			if ((s->lim - s->tok) > abs(s->tok - buf)) memmove(buf, s->tok, s->lim - s->tok);
			else memcpy(buf, s->tok, s->lim - s->tok);
			s->tok = buf;
			s->ptr = &buf[s->ptr - s->bot];
			cursor = &buf[cursor - s->bot];
			s->pos = &buf[s->pos - s->bot];
			s->lim = &buf[s->lim - s->bot];
			s->top = &s->lim[BSIZE];
			free(s->bot);
			s->bot = buf;
		}
		cnt = fread((char*) s->lim, 1, BSIZE, s->fd);
		if(cnt != BSIZE) { 
			s->eof = &s->lim[cnt];
			*(s->eof)++ = '\n';
		}
		s->lim += cnt;
		//assert(cnt >= numtoread);
	}
	return cursor;
}

/*
uchar *fill(Scanner *s, uchar *cursor, int numtoread) {
	
	static bool firstcall = true;

	if (firstcall) {
		
		void *buf;
		struct stat statbuf;
		
		int ret = fstat(fileno(s->fd), &statbuf);
		buf = mmap(NULL, statbuf.st_size, PROT_READ, MAP_PRIVATE, fileno(s->fd), 0);
		assert(buf != NULL);
		s->lim = (uchar*) buf;
		s->lim += statbuf.st_size;
		s->eof = s->lim;
		return (uchar*) buf;	   		
	}
	else {
		assert(0==1);
	}
}
*/

int scan(Scanner *s) {

	//uchar *cursor = s->cur;

	static uchar *cursor = NULL;
	Item item;
	int state;

	for (;;) {

s_0:

	if ((cursor == s->eof) and (cursor != NULL)) {
		fprintf(stderr, "Reached the end of the inputfile - did you forget a closing bracket?\n");
		return s->rank;
	}
	
	s->tok = cursor;
	state = 0;
	
	// printf("Entered s_0\n");
	// if (((s->line % 100) == 0) or (s->line < 100)) printf("Line: %i\n", s->line);

	item.type = Undefined;
	item.label1 = NULL;
	item.label2 = NULL;
	item.cpu = 0;
	item.nic = 0;
	item.tag = 0;

/*!re2c
	re2c:indent:top = 2;
	
	NL			= "\r"? "\n" ;
	WS			= [ \t]+ ;
	RANK        = "rank" ;
	NUMRANKS	= "num_ranks" ;
	BROPEN      = "{" ;
	BRCLOSE     = "}" ;
	SEND		= "send" ;
	RECV		= "recv" ;
	CALC		= "calc" ;
	REQ			= "requires" ;
	IREQ		= "irequires" ;
	NIC			= "nic" ;
	CPU			= "cpu" ;
	TAG			= "tag" ;
	TO			= "to" ;
	FROM		= "from" ;
	COLON		= ":" ;
	BYTE		= "b" ;
	LCOMMENT    = "//" ;
	OCOMMENT    = "/*" ;
	CCOMMENT    = "*/" ;
	ANYSOURCE   = "-1" ;
	ANYTAG      = "-1" ;
	INT         = [0-9]+ ; 
	IDENT       = [a-zA-Z][a-zA-Z0-9_]* ;
	ANY			= [^] ;

	WS			{ s->tok = cursor; }
	SEND		{ item.type = SendOp;  goto s_2; }
	RECV		{ item.type = RecvOp;  goto s_2; }
	CALC		{ item.type = LoclOp;  goto s_3; }
	RANK        { goto s_20; }
	NUMRANKS	{ goto s_22; }
	IDENT		{ item.label1 = add_label(s->tok, cursor); goto s_1; }
	LCOMMENT    { goto s_23; }
	OCOMMENT    { goto s_24; }

	NL			{ s->line++; continue; }
	BRCLOSE     { if (s->rank == -1) goto s_err; int oldrank = s->rank; s->rank = -1; return oldrank; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_1:
	state =1;

	// printf("Entered s_1\n");

/*!re2c
	WS			{ goto s_1; }
	COLON		{ goto s_4; }
	REQ			{ item.type = Dependency;       goto s_5; }
	IREQ		{ item.type = StartDependency;  goto s_5; }
	ANY			{ goto s_err; }
*/
	assert(0==1); //We should never reach this line

s_2:
	state=2;

	// printf("Entered s_2\n");

		s->tok = cursor;
/*!re2c
	WS			{ goto s_2; }
	INT			{ item.size = add_number(s->tok, cursor); goto s_10; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_3:
	state = 3;
	// printf("Entered s_3\n");

		s->tok = cursor;
/*!re2c
	WS			{ goto s_3; }
	INT			{ item.size = add_number(s->tok, cursor); goto s_7; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_4:
	state = 4;
	// printf("Entered s_4\n");

/*!re2c
	WS			{ goto s_4; }
	CALC		{ item.type = LoclOp; goto s_3; }
	SEND		{ item.type = SendOp; goto s_2; }
	RECV		{ item.type = RecvOp; goto s_2; }
	ANY			{ goto s_err; }
*/
	assert(0==1); //We should never reach this line

s_5:
	state =5;
	// printf("Entered s_5\n");

		s->tok = cursor;
/*!re2c
	WS			{ goto s_5; }
	IDENT		{ item.label2 = add_label(s->tok, cursor); goto s_6; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_6:
	state = 6;
	// printf("Entered s_6\n");

/*!re2c
	WS			{ goto s_6; }
	NL			{ s->line++; process_item(s, &item); continue; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_7:
	state = 7;
	// printf("Entered s_7\n");

/*!re2c
	WS			{ goto s_7; }
	CPU			{ goto s_8; }
	NL			{ s->line++; process_item(s, &item); continue; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_8:
	state =8;
	// printf("Entered s_8\n");
	
	s->tok = cursor;

/*!re2c
	WS			{ goto s_8; }
	INT			{ item.cpu = add_number(s->tok, cursor); goto s_9; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_9:
	state=9;
	// printf("Entered s_9\n");

/*!re2c
	WS			{ goto s_9; }
	NL			{ s->line++; process_item(s, &item); continue; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_10:
	state=10;
	// printf("Entered s_10\n");

/*!re2c
	WS			{ goto s_10; }
	BYTE		{ goto s_11; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_11:
	state = 11;
	// printf("Entered s_11\n");

/*!re2c
	WS			{ goto s_11; }
	TO			{ if (item.type == SendOp) {goto s_12;} else {goto s_err;}; }
	FROM		{ if (item.type == RecvOp) {goto s_12;} else {goto s_err;}; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_12:
	state = 12;	
	// printf("Entered s_12\n");
	
	s->tok = cursor;

/*!re2c
	WS			{ goto s_12; }
	ANYSOURCE   {if (item.type == RecvOp) {item.target = std::numeric_limits<uint32_t>::max(); goto s_13;} else {goto s_err;}; }
	INT			{ item.target = add_number(s->tok, cursor); goto s_13; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_13:
	state = 13;
	// printf("Entered s_13\n");

/*!re2c
	WS			{ goto s_13; }
	TAG			{ goto s_14; }
	CPU			{ goto s_16; }
	NIC			{ goto s_18; }
	NL			{ s->line++; process_item(s, &item); continue; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_14:
	state = 14;
	// printf("Entered s_14\n");

	s->tok = cursor;

/*!re2c
	WS			{ goto s_14; }
	ANYTAG      { item.tag = std::numeric_limits<uint32_t>::max(); goto s_15; }
	INT			{ item.tag = add_number(s->tok, cursor); goto s_15; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_15:
	state =15;
	// printf("Entered s_15\n");

/*!re2c
	WS			{ goto s_15; }
	CPU			{ goto s_16; }
	NIC			{ goto s_18; }
	NL			{ s->line++; process_item(s, &item); continue; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_16:
	state = 16;
	// printf("Entered s_16\n");
	
	s->tok = cursor;

/*!re2c
	WS			{ goto s_16; }
	INT			{ item.cpu = add_number(s->tok, cursor); goto s_17; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_17:
	state = 17;
	// printf("Entered s_17\n");
	
/*!re2c
	WS			{ goto s_17; }
	NIC			{ goto s_18; }
	NL			{ s->line++; process_item(s, &item); continue; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_18:
	state = 18;
	// printf("Entered s_18\n");
	
	s->tok = cursor;

/*!re2c
	WS			{ goto s_18; }
	INT			{ item.nic = add_number(s->tok, cursor); goto s_19; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_19:
	state = 19;
	// printf("Entered s_19\n");

/*!re2c
	WS			{ goto s_19; }
	NL			{ s->line++; process_item(s, &item); continue; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_20:
	state = 20;
	// printf("Entered s_20\n");

	s->tok = cursor;

/*!re2c
	WS			{ goto s_20; }
	INT			{ s->rank = add_number(s->tok, cursor); s->curr_rank = s->rank; goto s_21; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_21:
	state = 21;
	// printf("Entered s_21\n");

/*!re2c
	WS			{ goto s_21; }
	BROPEN		{ goto s_0; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_22:
	state = 22;
	// printf("Entered s_22\n");

	s->tok = cursor;

/*!re2c
	WS			{ goto s_22; }
	INT			{ s->num_ranks = add_number(s->tok, cursor); goto s_0; }
	ANY			{ goto s_err; }
*/

	assert(0==1); //We should never reach this line

s_23:
	state = 23;
	// printf("Entered s_23\n");

/*!re2c
    NL          { s->line++; continue;  }
	ANY			{ goto s_23; }
*/

	assert(0==1); //We should never reach this line

s_24:
	state = 24;
	// printf("Entered s_24\n");

/*!re2c
	CCOMMENT    { continue; }
    NL          { s->line++; goto s_24;  }
	ANY			{ goto s_24; }
*/

	assert(0==1); //We should never reach this line
	
s_err:

	fprintf(stderr, "Error in line %i:\n", s->line);
	
	uchar* nlbef = s->bot;
	uchar* nlaft = s->lim;
	// find the last newline before the error
	uchar* c = cursor-2;
	while (c>s->bot) {
		c--;
		if (*c == '\n') {
			nlbef = c;
			break;
		}
	}
	// find the next newline after the error
	c = cursor-2;
	while (c<s->lim) {
		c++;
		if (*c == '\n') {
			nlaft = c;
			break;
		}
	}

	for (uchar* c = nlbef+1; c<nlaft; c++) {
		fprintf(stderr, "%c", *c);
	}
	fprintf(stderr, "\n");

	for (int cnt=0; cnt<(cursor-2)-nlbef; cnt++) {
		fprintf(stderr, " ");
	}
	fprintf(stderr, "^\n");


	switch(state) {
		case 0:
			fprintf(stderr,"Expected: \"rank\", \"num_ranks\", \"calc\", \"send\", \"recv\", an identifier, \"}\" or \"\\n\"\n");
			break;
		case 1:
			fprintf(stderr,"Expected: \":\", \"requires\" or \"irequires\"\n");
			break;
		case 2:
			fprintf(stderr,"Expected: an integer\n");
			break;
		case 3:
			fprintf(stderr,"Expected: an integer\n");
			break;
		case 4:
			fprintf(stderr,"Expected: \"calc\", \"send\" or \"recv\"\n");
			break;
		case 5:
			fprintf(stderr,"Expected: an identifier\n");
			break;
		case 6:
			fprintf(stderr,"Expected: \"\\n\"\n");
			break;
		case 7:
			fprintf(stderr,"Expected: \"cpu\" or \"\\n\"\n");
			break;
		case 8:
			fprintf(stderr,"Expected: an integer\n");
			break;
		case 9:
			fprintf(stderr,"Expected: \"\\n\"\n");
			break;
		case 10:
			fprintf(stderr,"Expected: \"b\"\n");
			break;
		case 11:
			if (item.type == SendOp) fprintf(stderr,"Expected: \"to\"\n");
			else if (item.type == RecvOp) fprintf(stderr,"Expected: \"from\"\n");
			else fprintf(stderr,"Expected: \"to\" or \"from\"\n");
			break;
		case 12:
			fprintf(stderr,"Expected: an integer\n");
			break;
		case 13:
			fprintf(stderr,"Expected: \"tag\", \"cpu\", \"nic\" or \"\\n\"\n");
			break;
		case 14:
			fprintf(stderr,"Expected: an integer\n");
			break;
		case 15:
			fprintf(stderr,"Expected: \"cpu\", \"nic\" or \"\\n\"\n");
			break;
		case 16:
			fprintf(stderr,"Expected: an integer\n");
			break;
		case 17:
			fprintf(stderr,"Expected: \"nic\" or \"\\n\"\n");
			break;
		case 18:
			fprintf(stderr,"Expected: an integer\n");
			break;
		case 19:
			fprintf(stderr,"Expected: \"\\n\"\n");
			break;
		case 20:
			fprintf(stderr,"Expected: an integer\n");
			break;
		case 21:
			fprintf(stderr,"Expected: \"{\"\n");
			break;
		case 22:
			fprintf(stderr,"Expected: an integer\n");
			break;
	}
	
	if (*(cursor-1) == '\n') {
		fprintf(stderr, "Instead the schedule contained: \"\\n\"\n");
	}
	else {
		fprintf(stderr, "Instead the schedule contained: \"%c\"\n", *(cursor-1));
	}


	
	exit(EXIT_FAILURE);

}
}

main(int argc, char **argv){
    
	Scanner in;
	int lastprogress = 0;

	if (cmdline_parser(argc, argv, &args_info) != 0) {
		fprintf(stderr, "Couldn't parse command line arguments!\n");
		exit(EXIT_FAILURE);
	}

    memset((char*) &in, 0, sizeof(in));
   	in.fd = fopen(args_info.input_arg, "r");
	if (in.fd == NULL) {
		fprintf(stderr, "Couldn't open input file %s!\n", args_info.input_arg);
		exit(EXIT_FAILURE);
	}
	in.idtbl = new std::map<std::string, goalop_t>;
    
	uint32_t numranks = -1;

	while (true) {
		
		in.schedule = new Goal;
		
		scan(&in);

		if (in.num_ranks < 1) {
			fprintf(stderr, "Parse error: Number of Ranks undefined\n");
			exit(EXIT_FAILURE);
		}

		in.idtbl->clear();

		in.schedule->SetRank(in.curr_rank);
		in.schedule->SetNumRanks(in.num_ranks);
		in.schedule->SerializeSchedule(args_info.output_arg);
		delete in.schedule;
		int newprogress = round((((double) in.curr_rank) / in.num_ranks)*100);
		if (args_info.progress_given && (newprogress > lastprogress) ) {
			lastprogress = newprogress;
			printf("Progress %i%% - parsed schedule %i/%i\n", lastprogress, in.curr_rank, in.num_ranks);
		}
		if (in.curr_rank+1 == in.num_ranks) break;
	}
	
	free(in.bot);
	exit(EXIT_SUCCESS);
}
