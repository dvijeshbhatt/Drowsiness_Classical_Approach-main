// Problem with this:
/* Even if face is not detected by the famemesh still it is giving us all 468 point's values based
on previous frame. So, EAR will be the same for all consecutive frames.

Solution:
Return some values where the face is not detecting, lets assume that we are passing -1 as value.

Second:
For handing the missing data, prevoious frame EAR should be used. Reason is that, if we will use
future frame values and takes the average of them, then it will take some time which may delay 
sleep and microsleep detection.
 */

#include<bits/stdc++.h>
#include<iostream>
#include "Sleep_Msleep_module.cpp"

using namespace std;

class Missing_EAR_frames{
    public:
    float Current_frame_EAR, PreviousFrame_EAR;
    
    /*void init(){
        Current_frame_EAR = 0.0;
        PreviousFrame_EAR = 0.0;
    }*/

    float missing_EAR_frame(float EAR){
        if (EAR == -1){
            Current_frame_EAR = PreviousFrame_EAR;
        }
        else{
            Current_frame_EAR = EAR;
            PreviousFrame_EAR = EAR;
        }
        return Current_frame_EAR;
    }
};

int main(){
    Missing_EAR_frames mef;
    Sleep_Msleep obj;
    //mef.init();
    vector<pair<int, float>> FEAR_vect; // pair of Frame no and EAR
    FEAR_vect.push_back(make_pair(1, 0.2));
    FEAR_vect.push_back(make_pair(2, 0.1));
    FEAR_vect.push_back(make_pair(3, 0.04));
    FEAR_vect.push_back(make_pair(4, -1));
    FEAR_vect.push_back(make_pair(5, -1));
    FEAR_vect.push_back(make_pair(6, -1));
    FEAR_vect.push_back(make_pair(7, 0.1));
    FEAR_vect.push_back(make_pair(8, 0.2));
    FEAR_vect.push_back(make_pair(9, 0.13));
    FEAR_vect.push_back(make_pair(10, 0.14));
    FEAR_vect.push_back(make_pair(11, 0.05));
    FEAR_vect.push_back(make_pair(12, 0.02));
    FEAR_vect.push_back(make_pair(13, 0.01));
    FEAR_vect.push_back(make_pair(14, 0.04));
    FEAR_vect.push_back(make_pair(15, 0.06));
    FEAR_vect.push_back(make_pair(16, 0.08));
    FEAR_vect.push_back(make_pair(17, 0.07));
    FEAR_vect.push_back(make_pair(18, 0.06));
    FEAR_vect.push_back(make_pair(19, 0.05));
    FEAR_vect.push_back(make_pair(20, 0.04));
    FEAR_vect.push_back(make_pair(21, 0.03));
    FEAR_vect.push_back(make_pair(22, 0.02));
    FEAR_vect.push_back(make_pair(23, 0.01));
    FEAR_vect.push_back(make_pair(24, 0.12));
    FEAR_vect.push_back(make_pair(25, 0.02));
    FEAR_vect.push_back(make_pair(26, 0.04));
    FEAR_vect.push_back(make_pair(27, 0.15));
    FEAR_vect.push_back(make_pair(28, 0.17));
    FEAR_vect.push_back(make_pair(29, 0.03));
    FEAR_vect.push_back(make_pair(30, 0.04));
    FEAR_vect.push_back(make_pair(31, 0.15));
    FEAR_vect.push_back(make_pair(32, 0.14));
    FEAR_vect.push_back(make_pair(33, 0.12));
    FEAR_vect.push_back(make_pair(34, 0.01));
    FEAR_vect.push_back(make_pair(35, 0.4));
    FEAR_vect.push_back(make_pair(36, 0.23));
    FEAR_vect.push_back(make_pair(37, 0.1));
    FEAR_vect.push_back(make_pair(38, 0.07));
    FEAR_vect.push_back(make_pair(39, 0.08));
    FEAR_vect.push_back(make_pair(40, 0.09));
    FEAR_vect.push_back(make_pair(41, 0.2));
    obj.init();
    int counter;
    //enum stat s1;
    string decision;
    
    for (int i = 0; i < FEAR_vect.size(); i++)
    {
        //cout << "Frame ID: " << FEAR_vect[i].first << " EAR: " << mef.missing_EAR_frame(FEAR_vect[i].second) << endl;
        FEAR_vect[i].second = mef.missing_EAR_frame(FEAR_vect[i].second);
        counter = obj.sleep_msleep_detection(FEAR_vect[i].second, FEAR_vect[i].first); //FEAR_vect[i].first);
        //s1 = obj.sleep_msleep_detection(FEAR_vect[i].second,s1);
        //cout << "Counter: " << counter << endl;
        decision = obj.decision(counter);
        cout << "Frame ID: " << FEAR_vect[i].first << " EAR: " << FEAR_vect[i].second << " Decision: " << decision << " \t Counter: "<<counter << endl;

    }
 }