// check the ear for last 25 frames and check out of that if outof 25 frames 
// 20 frames are closed then it is a microsleep

#include<iostream>
#include<bits/stdc++.h>
//#include<Missing_EAR_frames.cpp>
using namespace std;

class Sleep_Msleep{
    public:
    float EAR_Threshold,Sleep_counter_threshold, MSleep_counter_threshold;
    int Current_frame, PreviousFrame, SCounter,Totalfrcount;
    enum State {No_Sleep, ToSleep, Sleep, To_No_Sleep};

    void init(){
        EAR_Threshold = 0.1;
        Sleep_counter_threshold = 18;
        MSleep_counter_threshold = 10;
        Totalfrcount = 0;
        SCounter = 0;
        State s = No_Sleep;
    }

    State sleep_msleep_detection(float EAR, State s){
        //cout<<"FrameID: "<<FrameID<<" EAR: "<<EAR<<endl;

        switch (s)
        {
            case No_Sleep:
                if(EAR > EAR_Threshold){
                    Totalfrcount = 0; SCounter = 0;
                }
                else
                {
                    Totalfrcount = 1;
                    SCounter = 1;
                    s = ToSleep;
                }
                break;
            case ToSleep:
                if (EAR > EAR_Threshold){
                    Totalfrcount = 0; SCounter = 0;
                    s = No_Sleep;
                }
                else{
                    SCounter++;
                    Totalfrcount++;
                    s = Sleep;
                }
                break;
            case Sleep:
                if(EAR > EAR_Threshold)
                {
                    Totalfrcount ++;
                    s = To_No_Sleep;
                }
                else{
                    Totalfrcount++;
                    SCounter++;
                    s = Sleep;
                }
                break;
            case To_No_Sleep:
                if(EAR > EAR_Threshold)
                {
                    Totalfrcount ++;
                    SCounter ++;
                    s = No_Sleep;
                }
                else{
                    Totalfrcount++;SCounter++;
                    s = Sleep;
                }
        }
        cout<<"Frame count : "<<Totalfrcount<<endl;
        if(Totalfrcount >=25)
        {
            if (SCounter >=23)
                cout<<"Alert!!! Driver is sleepy"<<endl;
            else
                cout<<"Driver is alert"<<endl;
        }
        return s;
    }       
};
