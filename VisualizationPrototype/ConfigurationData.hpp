#ifndef CONFIGURATIONDATA_H_
#define CONFIGURATIONDATA_H_

class ObjectData {
public:
	float radius, isoValue;
	float force[3];
};

class ConfigurationData {
public:
	ConfigurationData() {
		for (int i=0; i<20; i++) {
			objectData[i].radius=1.0f;
			objectData[i].isoValue=0.5f;
			objectData[i].force[0]=0.0f;
			objectData[i].force[1]=0.0f;
			objectData[i].force[2]=0.0f;
		}

	}

	virtual ~ConfigurationData() {

	}

	ObjectData objectData[20];
};

#endif /* CONFIGURATIONDATA_H_ */
