// check the ear for last 25 frames and check out of that if outof 25 frames 
// 20 frames are closed then it is a microsleep

#include<iostream>
#include<bits/stdc++.h>
using namespace std;

class Sleep_Msleep{
    public:
    float EAR_Threshold,Sleep_counter_threshold, MSleep_counter_threshold;
    int SCounter, Errorcounter, ErrorThreshold;

    void init(){
        EAR_Threshold = 0.1;
        Sleep_counter_threshold = 30;
        MSleep_counter_threshold = 10;
        SCounter = 0;
        Errorcounter = 0;
        ErrorThreshold=0;
    }

    int sleep_msleep_detection(float EAR , int FrameID){
        if (EAR < EAR_Threshold){
            if (SCounter == 0){
                SCounter = 1;
            }
            else{
                SCounter = SCounter + 1;
            }
        }
        else{
            if(SCounter > 0)
            {
                Errorcounter ++;
                //cout<<"Errorcounter : " << Errorcounter<<endl;
                SCounter++;
            }
            if(SCounter<=MSleep_counter_threshold)
                ErrorThreshold = 2;  // FPS * 0.1 * 1
            else
                ErrorThreshold = 6;  // FPS * 0.1 * 3
            if(Errorcounter>ErrorThreshold){
                SCounter = 0;
                Errorcounter = 0;
            }
        }
        return SCounter;
    }
    string decision(int counter){
        if (counter >= Sleep_counter_threshold){
            return "Alert!!!! Driver State : Sleep \t take a break";
        }
        else if (counter >= MSleep_counter_threshold){
            return "Alert!!!! Driver State : MicroSleep";
        }
        else{
            return "Drive State : Awake";
        }
    }
};
