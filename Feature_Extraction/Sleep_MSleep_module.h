#include<iostream>
#include<bits/stdc++.h>
//#include<Missing_EAR_frames.cpp>
using namespace std;

float EAR_Threshold,Sleep_counter_threshold, MSleep_counter_threshold;
int Current_frame, PreviousFrame, SCounter;

    void init(){
        EAR_Threshold = 0.1;
        Sleep_counter_threshold = 18;
        MSleep_counter_threshold = 10;
        Current_frame = 0;
        PreviousFrame = 0;
        SCounter = 0;
    }

    int sleep_msleep_detection(float EAR , int FrameID){
        if (EAR < EAR_Threshold){
            Current_frame = FrameID;
            if (SCounter == 0){
                PreviousFrame = Current_frame;
                SCounter = 1;
            }
            else{
                if (Current_frame - PreviousFrame == 1){
                    PreviousFrame = Current_frame;
                    SCounter = SCounter + 1;
                }
                else{
                    SCounter = 0;
                }
            }
        }
        else{
            SCounter = 0;
        }
        return SCounter;
    }
    string decision(int counter){
        if (counter > Sleep_counter_threshold){
            return "Take a break as driver's state is Sleep";
        }
        else if (counter > MSleep_counter_threshold){
            return "Alert!!!! Driver State is : MicroSleep";
        }
        else{
            return "Drive State is : Awake";
        }
    }