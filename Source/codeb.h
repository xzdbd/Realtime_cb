#ifndef YUV_CB

#define CHANNELS 3

typedef struct ce {
	uchar learnHigh[CHANNELS];
	uchar learnLow[CHANNELS];
	uchar max[CHANNELS];
	uchar min[CHANNELS];
	float avg[CHANNELS];
	int f;
	int stale;
	int t_first_update;
	int t_last_update;

} code_element;

typedef struct code_book {
	code_element **cb;
	int numEntries;
	int t;
} codeBook;

void connected_Components(IplImage *mask, int poly1_hull0=1, float perimScale=4.0, int *num=NULL, CvRect *bbs=NULL, CvPoint *centers=NULL);

#endif
