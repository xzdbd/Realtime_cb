#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include "cvaux.h"
#include "codeb.h"

int CVCONTOUR_APPROX_LEVEL = 2;   
int CVCLOSE_ITR = 1;
				
#define CV_CVX_WHITE	CV_RGB(0xff,0xff,0xff)
#define CV_CVX_BLACK	CV_RGB(0x00,0x00,0x00)

codeBook* cA;
codeBook* cC;
codeBook* cD;  
int maxMod[CHANNELS];	
int minMod[CHANNELS]; 	
unsigned cbBounds[CHANNELS]; 
bool ch[CHANNELS];		
int nChannels = CHANNELS;
int imageLen = 0;
uchar *pColor;
int Td; 
int Tadd; 
int Tdel; 
int T=50; 
int Fadd=35;			
int Tavgstale=50;		
int Fd=2;				
int Tavgstale_cD=50;	
int fgcount=0;
float beta=0.1f;
float gamma=0.1f;
float forgratio=0.0f;
float Tadap_update=0.4f;

int clear_stale_entries(codeBook &c);
uchar background_Diff(uchar *p, codeBook &c, int numChannels, int *minMod, int *maxMod);
int update_codebook_model(uchar* p,codeBook& c,unsigned* cbBounds,int numChannels);              
int trainig_codebook(uchar* p,codeBook& c,unsigned* cbBounds,int numChannels);
int training_clear_stale_entries(codeBook &c);
int det_update_codebook_cC(uchar* p,codeBook& c,unsigned* cbBounds,int numChannels);
int det_update_codebook_cD(uchar* p,codeBook& c,unsigned* cbBounds,int numChannels, int numframe); 
int realtime_clear_stale_entries_cC(codeBook &c, int FrmNum);
int realtime_clear_stale_entries_cD(codeBook &c, int FrmNum);
int cD_to_cC(codeBook &d, codeBook &c, int FrmNum);
uchar background_diff_realtime(uchar* p,codeBook& c,int numChannels,int* minMod,int* maxMod);


void help() {
	printf(
		"***Keep the focus on the video windows, NOT the consol***\n"
		"INTERACTIVE PARAMETERS:\n"
		"\tESC,q,Q  - quit the program\n"
		"\th	- print this help\n"
		"\tp	- pause toggle\n"
		"\ts	- single step\n"
		"\tr	- run mode (single step off)\n"
		"=== CODEBOOK PARAMS ===\n"
		"\ty,u,v- only adjust channel 0(y) or 1(u) or 2(v) respectively\n"
		"\ta	- adjust all 3 channels at once\n"
		"\tb	- adjust both 2 and 3 at once\n"
		"\ti,o	- bump upper threshold up,down by 1\n"
		"\tk,l	- bump lower threshold up,down by 1\n"
        "\tz,x     - bump Fadd threshold up,down by 1\n"
		"\tn,m     - bump Tavgstale threshold up,down by 1\n"
		"\t        Fadd小更新快,Tavgstale大更新快\n"
		);
}

int count_Segmentation(codeBook *c, IplImage *I, int numChannels, int *minMod, int *maxMod)
{
	int count = 0,i;
	uchar *pColor;
	int imageLen = I->width * I->height;

	//GET BASELINE NUMBER OF FG PIXELS FOR Iraw
	pColor = (uchar *)((I)->imageData);
	for(i=0; i<imageLen; i++)
	{
		if(background_Diff(pColor, c[i], numChannels, minMod, maxMod))
			count++;
		pColor += 3;
	}
    fgcount=count;
	return(fgcount);
}

void connected_Components(IplImage *mask, int poly1_hull0, float perimScale, int *num, CvRect *bbs, CvPoint *centers)
{
	static CvMemStorage*	mem_storage	= NULL;
	static CvSeq*			contours	= NULL;
	//CLEAN UP RAW MASK
	cvMorphologyEx( mask, mask, NULL, NULL, CV_MOP_OPEN, CVCLOSE_ITR );
	cvMorphologyEx( mask, mask, NULL, NULL, CV_MOP_CLOSE, CVCLOSE_ITR );

	//FIND CONTOURS AROUND ONLY BIGGER REGIONS
	if( mem_storage==NULL ) mem_storage = cvCreateMemStorage(0);
    else cvClearMemStorage(mem_storage);

	CvContourScanner scanner = cvStartFindContours(mask,mem_storage,sizeof(CvContour),CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
	CvSeq* c;
	int numCont = 0;
	while( (c = cvFindNextContour( scanner )) != NULL )
	{
		double len = cvContourPerimeter( c );
		double q = (mask->height + mask->width) /perimScale;   //calculate perimeter len threshold
		if( len < q ) //Get rid of blob if it's perimeter is too small
		{
			cvSubstituteContour( scanner, NULL );
		}
		else //Smooth it's edges if it's large enough
		{
			CvSeq* c_new;
			if(poly1_hull0) //Polygonal approximation of the segmentation
	            c_new = cvApproxPoly(c,sizeof(CvContour),mem_storage,CV_POLY_APPROX_DP, CVCONTOUR_APPROX_LEVEL,0);
			else //Convex Hull of the segmentation
				c_new = cvConvexHull2(c,mem_storage,CV_CLOCKWISE,1);
            cvSubstituteContour( scanner, c_new );
			numCont++;
        }
	}
	contours = cvEndFindContours( &scanner );

	// PAINT THE FOUND REGIONS BACK INTO THE IMAGE
	cvZero( mask );
	IplImage *maskTemp;
	//CALC CENTER OF MASS AND OR BOUNDING RECTANGLES
	if(num != NULL)
	{
		int N = *num, numFilled = 0, i=0;
		CvMoments moments;
		double M00, M01, M10;
		maskTemp = cvCloneImage(mask);
		for(i=0, c=contours; c != NULL; c = c->h_next,i++ )
		{
			if(i < N) //Only process up to *num of them
			{
				cvDrawContours(maskTemp,c,CV_CVX_WHITE, CV_CVX_WHITE,-1,CV_FILLED,8);
				//Find the center of each contour
				if(centers != NULL)
				{
					cvMoments(maskTemp,&moments,1);
					M00 = cvGetSpatialMoment(&moments,0,0);
					M10 = cvGetSpatialMoment(&moments,1,0);
					M01 = cvGetSpatialMoment(&moments,0,1);
					centers[i].x = (int)(M10/M00);
					centers[i].y = (int)(M01/M00);
				}
				//Bounding rectangles around blobs
				if(bbs != NULL)
				{
					bbs[i] = cvBoundingRect(c);
				}
				cvZero(maskTemp);
				numFilled++;
			}
			//Draw filled contours into mask
			cvDrawContours(mask,c,CV_CVX_WHITE,CV_CVX_WHITE,-1,CV_FILLED,8); //draw to central mask
		} //end looping over contours
		*num = numFilled;
		cvReleaseImage( &maskTemp);
	}
	else
	{
		for( c=contours; c != NULL; c = c->h_next )
		{
			cvDrawContours(mask,c,CV_CVX_WHITE, CV_CVX_BLACK,-1,CV_FILLED,8);
		}
	}
}

////////////////////////////
int main(int argc, char** argv)
{
	IplImage* temp1 = NULL;
    IplImage* temp2 = NULL;
    IplImage* result = NULL;
    IplImage* result1 = NULL;
    IplImage* result2 = NULL;

	CvBGStatModel* bg_model=0;
	CvBGStatModel* bg_model1=0;

    IplImage* rawImage = 0; 
	IplImage* yuvImage = 0; 
	IplImage* rawImage1 = 0;
    IplImage* pFrImg = 0;
	IplImage* pFrImg1= 0;
	IplImage* pFrImg2= 0;
	IplImage* ImaskCodeBookCC = 0;
    CvCapture* capture = 0;

	int c,n;

	maxMod[0] = 25; 
	minMod[0] = 35;
	maxMod[1] = 8;
	minMod[1] = 8;
	maxMod[2] = 8;
	minMod[2] = 8;

    argc=2;
    argv[1]="intelligentroom_raw.avi";
    if( argc > 2 )
	{
		fprintf(stderr, "Usage: bkgrd [video_file_name]\n");
		return -1;
	}
 
    if (argc ==1)
		if( !(capture = cvCaptureFromCAM(-1)))
		{
			fprintf(stderr, "Can not open camera.\n");
			return -2;
		}

    if(argc == 2)
		if( !(capture = cvCaptureFromFile(argv[1])))
		{
			fprintf(stderr, "Can not open video file %s\n", argv[1]);
			return -2;
		}

	bool pause = false;
	bool singlestep = false;

    if( capture )
    {
        cvNamedWindow( "原视频序列图像", 1 );
		cvNamedWindow("不实时更新的Codebook算法[本文]",1);
		cvNamedWindow("实时更新的Codebook算法[本文]",1);
		cvNamedWindow("基于MOG的方法[Chris Stauffer'2001]",1);
		cvNamedWindow("三帧差分", 1);
		cvNamedWindow("基于Bayes decision的方法[Liyuan Li'2003]", 1);
        
		cvMoveWindow("原视频序列图像", 0, 0);
		cvMoveWindow("不实时更新的Codebook算法[本文]", 360, 0);
		cvMoveWindow("实时更新的Codebook算法[本文]", 720, 350);
		cvMoveWindow("基于MOG的方法[Chris Stauffer'2001]", 0, 350);
		cvMoveWindow("三帧差分", 720, 0);
        cvMoveWindow("基于Bayes decision的方法[Liyuan Li'2003]",360, 350);
        int nFrmNum = -1;
        for(;;)
        {
    		if(!pause)
			{
				rawImage = cvQueryFrame( capture );
				++nFrmNum;
				printf("第%d帧\n",nFrmNum);
				if(!rawImage) 
					break;
			}
			if(singlestep)
			{
				pause = true;
			}
			if(0 == nFrmNum) 
			{
				printf(". . . wait for it . . .\n"); 
				
				temp1 = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 3);
				temp2 = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 3);
				result1 = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
				result2 = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
				result = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);

				bg_model = cvCreateGaussianBGModel(rawImage);
                bg_model1 = cvCreateFGDStatModel(rawImage);
				rawImage1 = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 3 );

				yuvImage = cvCloneImage(rawImage);
				pFrImg  = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
				pFrImg1 = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
				pFrImg2 = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );
				ImaskCodeBookCC = cvCreateImage( cvGetSize(rawImage), IPL_DEPTH_8U, 1 );

				imageLen = rawImage->width*rawImage->height;

				cA = new codeBook [imageLen];
				cC = new codeBook [imageLen];
				cD = new codeBook [imageLen];

				for(int f = 0; f<imageLen; f++)
				{
					cA[f].numEntries = 0; cA[f].t = 0;
					cC[f].numEntries = 0; cC[f].t = 0;
					cD[f].numEntries = 0; cD[f].t = 0;
				}
				for(int nc=0; nc<nChannels;nc++)
				{
					cbBounds[nc] = 10;
				}
				ch[0] = true;
				ch[1] = true;
				ch[2] = true;
			}
             
        	if( rawImage )
        	{
				if(!pause)
				{					
					cvSmooth(rawImage, rawImage1, CV_GAUSSIAN,3,3);

					cvChangeDetection(temp1, temp2, result1);
					cvChangeDetection(rawImage1, temp1, result2);
					cvAnd(result1, result2, result, NULL);
					cvCopy(temp1,temp2, NULL);
					cvCopy(rawImage,temp1, NULL);

					
					cvUpdateBGStatModel( rawImage, bg_model );
					cvUpdateBGStatModel( rawImage, bg_model1 );
				}

				cvCvtColor( rawImage1, yuvImage, CV_BGR2YCrCb );
				if( !pause && nFrmNum >= 1 && nFrmNum < T  )
				{
					pColor = (uchar *)((yuvImage)->imageData);
					for(int c=0; c<imageLen; c++)
					{
						update_codebook_model(pColor, cA[c],cbBounds,nChannels);
					    trainig_codebook(pColor, cC[c],cbBounds,nChannels);
						pColor += 3;
					}
				}

				if( nFrmNum == T)
				{
					for(c=0; c<imageLen; c++)
					{
						clear_stale_entries(cA[c]);
						training_clear_stale_entries(cC[c]);
					}
				}

				if(nFrmNum > T) 
				{
					pColor = (uchar *)((yuvImage)->imageData);
					uchar maskPixelCodeBook;
					uchar maskPixelCodeBook1;
					uchar maskPixelCodeBook2;
					uchar *pMask = (uchar *)((pFrImg)->imageData);
					uchar *pMask1 = (uchar *)((pFrImg1)->imageData);
					uchar *pMask2 = (uchar *)((pFrImg2)->imageData);
					for(int c=0; c<imageLen; c++)
					{
						//本文中不带自动背景更新的算法输出
						maskPixelCodeBook1=background_Diff(pColor, cA[c],nChannels,minMod,maxMod);
                        *pMask1++ = maskPixelCodeBook1;
						
						//本文中带自动背景更新的算法输出
						if ( !pause && det_update_codebook_cC(pColor, cC[c],cbBounds,nChannels))
						{	
							det_update_codebook_cD(pColor, cD[c],cbBounds,nChannels, nFrmNum); 
							realtime_clear_stale_entries_cD(cD[c], nFrmNum);
							cD_to_cC(cD[c], cC[c], (nFrmNum - T)/5);
							
						}
						else
						{
							realtime_clear_stale_entries_cC(cC[c], nFrmNum);
						
						} 

						maskPixelCodeBook2=background_Diff(pColor, cC[c],nChannels,minMod,maxMod);
						*pMask2++ = maskPixelCodeBook2;  
						pColor += 3;
					}

					cvCopy(pFrImg2,ImaskCodeBookCC);
					if(!pause)
					{
						count_Segmentation(cC,yuvImage,nChannels,minMod,maxMod);
						forgratio = (float) (fgcount)/ imageLen;
					}
				}
				bg_model1->foreground->origin=1;
				bg_model->foreground->origin=1;				
				pFrImg->origin=1;
                pFrImg1->origin=1;
				pFrImg2->origin=1;
				ImaskCodeBookCC->origin=1;
				result->origin=1;
				//connected_Components(pFrImg1,1,40);
				//connected_Components(pFrImg2,1,40);
				
                cvShowImage("基于MOG的方法[Chris Stauffer'2001]", bg_model->foreground);
           		cvShowImage( "原视频序列图像", rawImage );
				cvShowImage("三帧差分", result);
 				cvShowImage( "不实时更新的Codebook算法[本文]",pFrImg1);
				cvShowImage("实时更新的Codebook算法[本文]",pFrImg2);
				cvShowImage("基于Bayes decision的方法[Liyuan Li'2003]", bg_model1->foreground);

	         	c = cvWaitKey(1)&0xFF;
				//End processing on ESC, q or Q
				if(c == 27 || c == 'q' || c == 'Q')
					break;
				//Else check for user input
				switch(c)
				{
					case 'h':
						help();
						break;
					case 'p':
						pause ^= 1;
						break;
					case 's':
						singlestep = 1;
						pause = false;
						break;
					case 'r':
						pause = false;
						singlestep = false;
						break;
				//CODEBOOK PARAMS
                case 'y':
                case '0':
                        ch[0] = 1;
                        ch[1] = 0;
                        ch[2] = 0;
                        printf("CodeBook YUV Channels active: ");
                        for(n=0; n<nChannels; n++)
                                printf("%d, ",ch[n]);
                        printf("\n");
                        break;
                case 'u':
                case '1':
                        ch[0] = 0;
                        ch[1] = 1;
                        ch[2] = 0;
                        printf("CodeBook YUV Channels active: ");
                        for(n=0; n<nChannels; n++)
                                printf("%d, ",ch[n]);
                        printf("\n");
                        break;
                case 'v':
                case '2':
                        ch[0] = 0;
                        ch[1] = 0;
                        ch[2] = 1;
                        printf("CodeBook YUV Channels active: ");
                        for(n=0; n<nChannels; n++)
                                printf("%d, ",ch[n]);
                        printf("\n");
                        break;
                case 'a': //All
                case '3':
                        ch[0] = 1;
                        ch[1] = 1;
                        ch[2] = 1;
                        printf("CodeBook YUV Channels active: ");
                        for(n=0; n<nChannels; n++)
                                printf("%d, ",ch[n]);
                        printf("\n");
                        break;
                case 'b':  //both u and v together
                        ch[0] = 0;
                        ch[1] = 1;
                        ch[2] = 1;
                        printf("CodeBook YUV Channels active: ");
                        for(n=0; n<nChannels; n++)
                                printf("%d, ",ch[n]);
                        printf("\n");
                        break;
                case 'z': 
					printf(" Fadd加1 ");
					Fadd += 1;
					printf("Fadd=%.4d\n",Fadd);										
					break;
				case 'x':
					printf(" Fadd减1 "); 
					Fadd -= 1;					
					printf("Fadd=%.4d\n",Fadd);										
					break;
				case 'n': 
					printf(" Tavgstale加1 ");
					Tavgstale += 1;
					printf("Tavgstale=%.4d\n",Tavgstale);										
					break;
				case 'm': 
					printf(" Tavgstale减1 ");
					Tavgstale -= 1;
					printf("Tavgstale=%.4d\n",Tavgstale);										
					break;
				case 'i': //modify max classification bounds (max bound goes higher)
					for(n=0; n<nChannels; n++)
					{
						if(ch[n])
							maxMod[n] += 1;
						printf("%.4d,",maxMod[n]);
					}
					printf(" CodeBook High Side\n");
					break;
				case 'o': //modify max classification bounds (max bound goes lower)
					for(n=0; n<nChannels; n++)
					{
						if(ch[n])
							maxMod[n] -= 1;
						printf("%.4d,",maxMod[n]);
					}
					printf(" CodeBook High Side\n");
					break;
				case 'k': //modify min classification bounds (min bound goes lower)
					for(n=0; n<nChannels; n++)
					{
						if(ch[n])
							minMod[n] += 1;
						printf("%.4d,",minMod[n]);
					}
					printf(" CodeBook Low Side\n");
					break;
				case 'l': //modify min classification bounds (min bound goes higher)
					for(n=0; n<nChannels; n++)
					{
						if(ch[n])
							minMod[n] -= 1;
						printf("%.4d,",minMod[n]);
					}
					printf(" CodeBook Low Side\n");
					break;
				}
            }
		}		
		cvReleaseCapture( &capture );
		cvReleaseBGStatModel((CvBGStatModel**)&bg_model);
		cvReleaseBGStatModel((CvBGStatModel**)&bg_model1);

        cvDestroyWindow( "原视频序列图像" );
		cvDestroyWindow( "不实时更新的Codebook算法[本文]");
		cvDestroyWindow( "实时更新的Codebook算法[本文]");
		cvDestroyWindow( "基于MOG的方法[Chris Stauffer'2001]");
		cvDestroyWindow( "三帧差分" );
		cvDestroyWindow( "基于Bayes decision的方法[Liyuan Li'2003]");

		cvReleaseImage(&temp1);
		cvReleaseImage(&temp2);
		cvReleaseImage(&result);
		cvReleaseImage(&result1);
		cvReleaseImage(&result2);
		cvReleaseImage(&pFrImg);
		cvReleaseImage(&pFrImg1);
		cvReleaseImage(&pFrImg2);

		if(yuvImage) cvReleaseImage(&yuvImage);
		if(rawImage) cvReleaseImage(&rawImage);
		if(rawImage1) cvReleaseImage(&rawImage1);
		if(ImaskCodeBookCC) cvReleaseImage(&ImaskCodeBookCC);
		delete [] cA;
		delete [] cC;
		delete [] cD;
    }
	else
	{ 
		printf("\n\nDarn, Something wrong with the parameters\n\n"); help();
	}
    return 0;
}

int clear_stale_entries(codeBook &c)
{
   int staleThresh = c.t>>1;
   int *keep = new int [c.numEntries];
   int keepCnt = 0;

   for(int i=0; i<c.numEntries; i++)
   {
      if(c.cb[i]->stale > staleThresh)
         keep[i] = 0;
      else
      {
         keep[i] = 1; 
         keepCnt += 1;
      }
   }
   c.t = 0;    
   code_element **foo = new code_element* [keepCnt];
   int k=0;
   for(int ii=0; ii<c.numEntries; ii++)
   {
      if(keep[ii])
      {
         foo[k] = c.cb[ii];
         foo[k]->t_last_update = 0;
         k++;
      }
   }
   delete [] keep;
   delete [] c.cb;
   c.cb = foo;
   int numCleared = c.numEntries - keepCnt;
   c.numEntries = keepCnt;
   return(numCleared);
}

uchar background_Diff(uchar *p, codeBook &c, int numChannels, int *minMod, int *maxMod)
{
	int matchChannel;
	int i;
	for(i=0; i<c.numEntries; i++)
	{
		matchChannel = 0;
		for(int n=0; n<numChannels; n++)
		{
			if((c.cb[i]->min[n] - minMod[n] <= *(p+n)) && (*(p+n) <= c.cb[i]->max[n] + maxMod[n]))
			{
				matchChannel++;
			}
			else
			{
				break;
			}
		}
		if(matchChannel == numChannels)
		{
			break;
		}
	}
	if(i >= c.numEntries) return(255);
	return(0);
}

int update_codebook_model(uchar* p,codeBook& c,unsigned* cbBounds,int numChannels)              
{
	if(c.numEntries == 0) c.t = 0;
	c.t += 1;
	unsigned int high[3],low[3];
	
	int matchChannel; 
	float avg[3];
	
	for(int i=0; i<c.numEntries; i++)
	{
		matchChannel = 0;
		for(int n=0; n<numChannels; n++)
		{
			if((c.cb[i]->learnLow[n] <= *(p+n)) && (*(p+n) <= c.cb[i]->learnHigh[n]))
			{
				matchChannel++;
			}
		}
		if(matchChannel == numChannels)
		{
			for(n=0; n<numChannels; n++)
			{
				avg[n] = (c.cb[i]->f * c.cb[i]->avg[n] + *(p+n))/(c.cb[i]->f + 1);
				c.cb[i]->avg[n] = avg[n];
				
				if(c.cb[i]->max[n] < *(p+n))
				{
					c.cb[i]->max[n] = *(p+n);
				}
				else if(c.cb[i]->min[n] > *(p+n))
				{
					c.cb[i]->min[n] = *(p+n);
				}
			}
			c.cb[i]->f += 1;

			c.cb[i]->t_last_update = c.t;
			int negRun = c.t - c.cb[i]->t_last_update;
			if(c.cb[i]->stale < negRun) c.cb[i]->stale = negRun;
			break;
		}
	}

	for(int n=0; n<numChannels; n++)
	{
		high[n] = *(p+n)+*(cbBounds+n);
		if(high[n] > 255) high[n] = 255;
		low[n] = *(p+n)-*(cbBounds+n);
		if(low[n] < 0) low[n] = 0;
	}
	if(i == c.numEntries)
	{
		code_element **foo = new code_element* [c.numEntries+1];
		for(int ii=0; ii<c.numEntries; ii++)
		{
			foo[ii] = c.cb[ii];
		}
		foo[c.numEntries] = new code_element;
		if(c.numEntries) delete [] c.cb;

		c.cb = foo;
		for(n=0; n<numChannels; n++) 
		{
			c.cb[c.numEntries]->avg[n] = *(p+n);
			c.cb[c.numEntries]->max[n] = *(p+n);
			c.cb[c.numEntries]->min[n] = *(p+n);
		
			c.cb[c.numEntries]->learnHigh[n] = high[n];
			c.cb[c.numEntries]->learnLow[n] = low[n];

		}
		c.cb[c.numEntries]->f = 1;
		c.cb[c.numEntries]->stale = c.t-1;
		c.cb[c.numEntries]->t_first_update = c.t;
		c.cb[c.numEntries]->t_last_update = c.t;		
		c.numEntries += 1;
	}

	for(int s=0; s<c.numEntries; s++)
	{
		int negRun = c.t - c.cb[s]->t_last_update + c.cb[s]->t_first_update -1 ;
		if(c.cb[s]->stale < negRun) c.cb[s]->stale = negRun;
		
	}

	for(n=0; n<numChannels; n++)
	{
		if(c.cb[i]->learnHigh[n] < high[n]) c.cb[i]->learnHigh[n] += 1;
		if(c.cb[i]->learnLow[n] > low[n]) c.cb[i]->learnLow[n] -= 1;
	}
	return(i);
}


int trainig_codebook(uchar* p,codeBook& c,unsigned* cbBounds,int numChannels)              
{
	if(c.numEntries == 0) c.t = 0;
	c.t += 1;
	unsigned int high[3],low[3];
	
	int matchChannel; 
	float avg[3];
	
	for(int i=0; i<c.numEntries; i++)
	{
		matchChannel = 0;
		for(int n=0; n<numChannels; n++)
		{
			if((c.cb[i]->learnLow[n] <= *(p+n)) && (*(p+n) <= c.cb[i]->learnHigh[n]))
			{
				matchChannel++;
			}
		}
		if(matchChannel == numChannels)
		{
			for(n=0; n<numChannels; n++)
			{
				avg[n] = (c.cb[i]->f * c.cb[i]->avg[n] + *(p+n))/(c.cb[i]->f + 1);
				c.cb[i]->avg[n] = avg[n];
				
				if(c.cb[i]->max[n] < *(p+n))
				{
					c.cb[i]->max[n] = *(p+n);
				}
				else if(c.cb[i]->min[n] > *(p+n))
				{
					c.cb[i]->min[n] = *(p+n);
				}
			}
			c.cb[i]->f += 1;

			c.cb[i]->t_last_update = c.t;
			int negRun = c.t - c.cb[i]->t_last_update;
			if(c.cb[i]->stale < negRun) c.cb[i]->stale = negRun;
			
			if (i!=0)
			{
				code_element **fo = new code_element* [c.numEntries];
				fo[0] = c.cb[i];
				for(int h=0; h<i; h++)
				{
					fo[h+1] = c.cb[h];
				}
				for(h=i+1; h<c.numEntries; h++)
				{
					fo[h] = c.cb[h];
				}
				if(c.numEntries) delete [] c.cb;
				c.cb = fo;
			}

			break;
		}
	}
	
	for(int n=0; n<numChannels; n++)
	{
		high[n] = *(p+n)+*(cbBounds+n);
		if(high[n] > 255) high[n] = 255;
		low[n] = *(p+n)-*(cbBounds+n);
		if(low[n] < 0) low[n] = 0;
	}
	if(i == c.numEntries)
	{
		code_element **foo = new code_element* [c.numEntries+1];
		for(int ii=0; ii<c.numEntries; ii++)
		{
			foo[ii] = c.cb[ii];
		}
		foo[c.numEntries] = new code_element;
		if(c.numEntries) delete [] c.cb;
		c.cb = foo;	
		for(n=0; n<numChannels; n++) 
		{
			c.cb[c.numEntries]->avg[n] = *(p+n);
			c.cb[c.numEntries]->max[n] = *(p+n);
			c.cb[c.numEntries]->min[n] = *(p+n);
			c.cb[c.numEntries]->learnHigh[n] = high[n];
			c.cb[c.numEntries]->learnLow[n] = low[n];

		}
		c.cb[c.numEntries]->f = 1;
		c.cb[c.numEntries]->stale = c.t-1;
		c.cb[c.numEntries]->t_first_update = c.t;
		c.cb[c.numEntries]->t_last_update = c.t;		
		c.numEntries += 1;
	}

	for(int s=0; s<c.numEntries; s++)
	{
		int negRun = c.t - c.cb[s]->t_last_update + c.cb[s]->t_first_update -1 ;
		if(c.cb[s]->stale < negRun) c.cb[s]->stale = negRun;
		
	}

	for(n=0; n<numChannels; n++)
	{
		if(c.cb[i]->learnHigh[n] < high[n]) c.cb[i]->learnHigh[n] += 1;
		if(c.cb[i]->learnLow[n] > low[n]) c.cb[i]->learnLow[n] -= 1;
	}
	return(i);
}

int training_clear_stale_entries(codeBook &c)
{
   int staleThresh = c.t>>1;
   int *keep = new int [c.numEntries];
   int keepCnt = 0;
   for(int i=0; i<c.numEntries; i++)
   {
      if(c.cb[i]->stale > staleThresh)
         keep[i] = 0;
      else
      {
         keep[i] = 1;
         keepCnt += 1;
      }
   }
   code_element **foo = new code_element* [keepCnt];
   int k=0;
   for(int ii=0; ii<c.numEntries; ii++)
   {
      if(keep[ii])
      {
         foo[k] = c.cb[ii];
         k++;
      }
   }

   delete [] keep;
   delete [] c.cb;
   c.cb = foo;
   int numCleared = c.numEntries - keepCnt;
   c.numEntries = keepCnt;
   return(numCleared);
}


int det_update_codebook_cC(uchar* p,codeBook& c,unsigned* cbBounds,int numChannels)              
{
	c.t += 1;

	int matchChannel; 
	float avg[3];
	int learnLow[3],learnHigh[3];

	for(int i=0; i<c.numEntries; i++)
	{
		matchChannel = 0;
		for(int n=0; n<numChannels; n++)
		{
			if (forgratio >= Tadap_update )
			{
				learnLow[n] = c.cb[i]->learnLow[n] * (1 - gamma);
				c.cb[i]->learnLow[n] = learnLow[n];
				learnHigh[n] = c.cb[i]->learnHigh[n] * (1 + gamma);
				c.cb[i]->learnHigh[n] = learnHigh[n];
			}
			if((c.cb[i]->learnLow[n] <= *(p+n)) && (*(p+n) <= c.cb[i]->learnHigh[n]))
			{
				matchChannel++;
			}
		}
		if(matchChannel == numChannels)
		{
			if (forgratio >= Tadap_update )
			{
				for(n=0; n<numChannels; n++)
				{
					avg[n] = (1 - beta) * c.cb[i]->avg[n] + *(p+n) * beta;
					c.cb[i]->avg[n] = avg[n];
				
					if(c.cb[i]->max[n] < *(p+n))
					{
						c.cb[i]->max[n] = *(p+n);
					}
					else if(c.cb[i]->min[n] > *(p+n))
					{
						c.cb[i]->min[n] = *(p+n);
					}
				}
			}
			else
			{
				for(n=0; n<numChannels; n++)
				{

					avg[n] = (c.cb[i]->f * c.cb[i]->avg[n] + *(p+n))/(c.cb[i]->f + 1);
					c.cb[i]->avg[n] = avg[n];
				
					if(c.cb[i]->max[n] < *(p+n))
					{
						c.cb[i]->max[n] = *(p+n);
					}
					else if(c.cb[i]->min[n] > *(p+n))
					{
						c.cb[i]->min[n] = *(p+n);
					}
				}
			}
			
			int negRun = c.t - c.cb[i]->t_last_update;
			if(c.cb[i]->stale < negRun) c.cb[i]->stale = negRun;
			c.cb[i]->t_last_update = c.t;
			c.cb[i]->f += 1;

			break;
			
		}
	}
	
	if(i == c.numEntries) return (i);
	return(0);

}


int det_update_codebook_cD(uchar* p,codeBook& c,unsigned* cbBounds,int numChannels, int numframe)              
{
	if(c.numEntries == 0) c.t = numframe -1;
	c.t += 1;

	unsigned int high[3],low[3];
	
	int matchChannel; 
	float avg[3];
	int learnLow[3],learnHigh[3];

	for(int i=0; i<c.numEntries; i++)
	{
		matchChannel = 0;
		for(int n=0; n<numChannels; n++)
		{
			if (forgratio >= Tadap_update )
			{
				learnLow[n] = c.cb[i]->learnLow[n] * (1 - gamma);
				c.cb[i]->learnLow[n] = learnLow[n];
				learnHigh[n] = c.cb[i]->learnHigh[n] * (1 + gamma);
				c.cb[i]->learnHigh[n] = learnHigh[n];
			}
			if((c.cb[i]->learnLow[n] <= *(p+n)) && (*(p+n) <= c.cb[i]->learnHigh[n]))
			{
				matchChannel++;
			}
		}
		if(matchChannel == numChannels)
		{

			if (forgratio >= Tadap_update )
			{
				for(n=0; n<numChannels; n++)
				{
					avg[n] = (1 - beta) * c.cb[i]->avg[n] + *(p+n) * beta;
					c.cb[i]->avg[n] = avg[n];
				
					if(c.cb[i]->max[n] < *(p+n))
					{
						c.cb[i]->max[n] = *(p+n);
					}
					else if(c.cb[i]->min[n] > *(p+n))
					{
						c.cb[i]->min[n] = *(p+n);
					}
				}
			}
			else
			{
				for(n=0; n<numChannels; n++)
				{

					avg[n] = (c.cb[i]->f * c.cb[i]->avg[n] + *(p+n))/(c.cb[i]->f + 1);
					c.cb[i]->avg[n] = avg[n];
				
					if(c.cb[i]->max[n] < *(p+n))
					{
						c.cb[i]->max[n] = *(p+n);
					}
					else if(c.cb[i]->min[n] > *(p+n))
					{
						c.cb[i]->min[n] = *(p+n);
					}
				}
			}
			int negRun = c.t - c.cb[i]->t_last_update;
			if(c.cb[i]->stale < negRun) c.cb[i]->stale = negRun;
			c.cb[i]->f += 1;
			c.cb[i]->t_last_update = c.t;
			break;
		}
	}
	for(int n=0; n<numChannels; n++)
	{
		high[n] = *(p+n)+*(cbBounds+n);
		if(high[n] > 255) high[n] = 255;
		low[n] = *(p+n)-*(cbBounds+n);
		if(low[n] < 0) low[n] = 0;
	}
	if(i == c.numEntries)
	{
		code_element **foo = new code_element* [c.numEntries+1];
		for(int ii=0; ii<c.numEntries; ii++)
		{
			foo[ii] = c.cb[ii];
		}
		foo[c.numEntries] = new code_element;
		if(c.numEntries) 
			delete [] c.cb;
		c.cb = foo;	
		for(n=0; n<numChannels; n++) 
		{
			c.cb[c.numEntries]->avg[n] = *(p+n);
			c.cb[c.numEntries]->max[n] = *(p+n);
			c.cb[c.numEntries]->min[n] = *(p+n);
		
			c.cb[c.numEntries]->learnHigh[n] = high[n];
			c.cb[c.numEntries]->learnLow[n] = low[n];

		}
		c.cb[c.numEntries]->f = 1;
		c.cb[c.numEntries]->stale = 0;
		c.cb[c.numEntries]->t_first_update = c.t;
		c.cb[c.numEntries]->t_last_update = c.t;		
		c.numEntries += 1;
	}

	for(int s=0; s<c.numEntries; s++)
	{
		int negRun = c.t - c.cb[s]->t_last_update;
		if(c.cb[s]->stale < negRun) c.cb[s]->stale = negRun;
		
	}

	for(n=0; n<numChannels; n++)
	{
		if(c.cb[i]->learnHigh[n] < high[n]) c.cb[i]->learnHigh[n] += 1;
		if(c.cb[i]->learnLow[n] > low[n]) c.cb[i]->learnLow[n] -= 1;
	}
	return(i);
}


int realtime_clear_stale_entries_cC(codeBook &c, int FrmNum)
{
	int staleThresh = FrmNum/2;	
	int *keep = new int [c.numEntries];
	int keepCnt = 0;

	for(int i=0; i<c.numEntries; i++)
	{
		if(c.cb[i]->stale > staleThresh)
			keep[i] = 0;
		else
		{
			keep[i] = 1;
			keepCnt += 1;
		}
	}
	c.t = 0;    
	code_element **foo = new code_element* [keepCnt];
	int k=0;
	for(int ii=0; ii<c.numEntries; ii++)
	{
		if(keep[ii])
		{
			foo[k] = c.cb[ii];
			k++;
		}
	}
	delete [] keep;
	delete [] c.cb;
	c.cb = foo;
	int numCleared = c.numEntries - keepCnt;
	c.numEntries = keepCnt;
	return(numCleared);
}

int realtime_clear_stale_entries_cD(codeBook &c, int FrmNum)
{
	int *keep = new int [c.numEntries];
	int keepCnt = 0;

	for(int i=0; i<c.numEntries; i++)
	{
		if(c.cb[i]->f <=Fd && c.cb[i]->stale >=Tavgstale_cD)
			keep[i] = 0;
		else
		{
			keep[i] = 1;
			keepCnt += 1;
		}
	}
  
	code_element **foo = new code_element* [keepCnt];
	int k=0;
	for(int ii=0; ii<c.numEntries; ii++)
	{
		if(keep[ii])
		{
			foo[k] = c.cb[ii];
			k++;
		}
	}
	delete [] keep;
	delete [] c.cb;
	c.cb = foo;
	int numCleared = c.numEntries - keepCnt;
	c.numEntries = keepCnt;
	return(numCleared);
}

int cD_to_cC(codeBook &d, codeBook &c, int FrmNum)
{
	int *keep_d = new int [d.numEntries];
	int keepCnt = 0;

	for(int i=0; i<d.numEntries; i++)
	{
		int convertThresh = (FrmNum - T)/d.cb[i]->f;
		if(d.cb[i]->f >=Fadd && convertThresh <=Tavgstale)
		{
			keep_d[i] = 0;
		}
		else
		{
			keep_d[i] = 1;
			keepCnt += 1;
		}
	}
 
	code_element **foo_d = new code_element* [keepCnt];
	int k=0;
	for(int ii=0; ii<d.numEntries; ii++)
	{
		if(keep_d[ii])
		{
			foo_d[k] = d.cb[ii];
			k++;
		}
		else
		{
			code_element **foo_c = new code_element* [c.numEntries+1];
			for(int jj=0; jj<c.numEntries; jj++)
			{
				foo_c[jj] = c.cb[jj];
			}
			foo_c[c.numEntries] = new code_element;

				delete [] c.cb;
			c.cb = foo_c;

			c.cb[c.numEntries] = d.cb[ii];
			c.numEntries +=1;
		}
		
	}
	delete [] keep_d;
	delete [] d.cb;
	d.cb = foo_d;
	int numconverted = d.numEntries - keepCnt;
	d.numEntries = keepCnt;
	return(numconverted);
}






