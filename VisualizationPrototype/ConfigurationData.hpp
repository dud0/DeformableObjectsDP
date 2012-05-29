#ifndef CONFIGURATIONDATA_H_
#define CONFIGURATIONDATA_H_

enum displayMode {NORMAL, TENSION, EDGE};

class ObjectData {
public:
	bool change;
	float radius, isoValue;
	float force[3];
	displayMode mode;
	float colorR, colorG, colorB, colorA;
};

class ConfigurationData {
public:
	ConfigurationData() {
		doPause = false;
		doRestart = false;

		for (int i=0; i<20; i++) {
			objectData[i].change = false;
			objectData[i].radius=1.0f;
			objectData[i].isoValue=0.5f;
			objectData[i].force[0]=0.0f;
			objectData[i].force[1]=0.0f;
			objectData[i].force[2]=0.0f;
			objectData[i].mode= NORMAL;
			objectData[i].colorR = 0.9f;
			objectData[i].colorG = 0.1f;
			objectData[i].colorB = 0.1f;
			objectData[i].colorA = 1.0f;
		}

		objectData[1].colorR = 0.1f;
		objectData[1].colorG = 0.9f;
		objectData[1].colorB = 0.1f;

		objectData[2].colorR = 0.1f;
		objectData[2].colorG = 0.1f;
		objectData[2].colorB = 0.9f;
	}

	virtual ~ConfigurationData() {

	}

	bool doPause, doRestart;
	ObjectData objectData[20];
};

#endif /* CONFIGURATIONDATA_H_ */
