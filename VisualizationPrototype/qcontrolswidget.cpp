#include "qcontrolswidget.h"
#include <stdio.h>

QControlsWidget::QControlsWidget(QWidget *parent, ConfigurationData * configData)
    : QWidget(parent)
{
	ui.setupUi(this);

	this->configData=configData;

	ui.comboBox->addItem("1");
	ui.comboBox->addItem("2");
	ui.comboBox->addItem("3");
	ui.comboBox->addItem("4");
	ui.comboBox->addItem("5");
	ui.comboBox->addItem("6");
	ui.comboBox->addItem("7");
	ui.comboBox->addItem("8");
	ui.comboBox->addItem("9");
	ui.comboBox->addItem("10");
	ui.comboBox->addItem("11");
	ui.comboBox->addItem("12");
	ui.comboBox->addItem("13");
	ui.comboBox->addItem("14");
	ui.comboBox->addItem("15");
	ui.comboBox->addItem("16");
	ui.comboBox->addItem("17");
	ui.comboBox->addItem("18");
	ui.comboBox->addItem("19");
	ui.comboBox->addItem("20");

	connect(ui.comboBox, SIGNAL(currentIndexChanged(int)),this, SLOT(showObjectData(int)));
	connect(ui.applyPushButton, SIGNAL(clicked()), this, SLOT(saveObjectData()));
	connect(ui.btnPause, SIGNAL(clicked()), this, SLOT(pause()));
	connect(ui.btnRestart, SIGNAL(clicked()), this, SLOT(restart()));

	showObjectData(0);
}

QControlsWidget::~QControlsWidget()
{

}

void QControlsWidget::showObjectData(int index) {
	ui.radiusLineEdit->setText(QString::number(configData->objectData[index].radius));
	ui.isovalueLineEdit->setText(QString::number(configData->objectData[index].isoValue));
	ui.xLineEdit->setText(QString::number(configData->objectData[index].force[0]));
	ui.yLineEdit->setText(QString::number(configData->objectData[index].force[1]));
	ui.zLineEdit->setText(QString::number(configData->objectData[index].force[2]));
	switch (configData->objectData[index].mode) {
	case NORMAL:
		ui.normalModeRadioButton->setChecked(true);
		break;
	case TENSION:
		ui.tensionModeRadioButton->setChecked(true);
		break;
	case EDGE:
		ui.edgeModeRadioButton->setChecked(true);
		break;
	}
	ui.rLineEdit->setText(QString::number(configData->objectData[index].colorR));
	ui.gLineEdit->setText(QString::number(configData->objectData[index].colorG));
	ui.bLineEdit->setText(QString::number(configData->objectData[index].colorB));
}

void QControlsWidget::saveObjectData() {
	configData->objectData[ui.comboBox->currentIndex()].change = true;
	int index = ui.comboBox->currentIndex();
	configData->objectData[index].radius=ui.radiusLineEdit->text().toFloat();
	configData->objectData[index].isoValue=ui.isovalueLineEdit->text().toFloat();
	configData->objectData[index].force[0]=ui.xLineEdit->text().toFloat();
	configData->objectData[index].force[1]=ui.yLineEdit->text().toFloat();
	configData->objectData[index].force[2]=ui.zLineEdit->text().toFloat();
	if (ui.normalModeRadioButton->isChecked()) {
		configData->objectData[index].mode=NORMAL;
	} else if (ui.tensionModeRadioButton->isChecked()) {
		configData->objectData[index].mode=TENSION;
	} else if (ui.edgeModeRadioButton->isChecked()) {
		configData->objectData[index].mode=EDGE;
	}
	configData->objectData[index].colorR=ui.rLineEdit->text().toFloat();
	configData->objectData[index].colorG=ui.gLineEdit->text().toFloat();
	configData->objectData[index].colorB=ui.bLineEdit->text().toFloat();
}

void QControlsWidget::restart() {
	configData->doRestart = true;
}

void QControlsWidget::pause() {
	if(configData->doPause == false) {
		configData->doPause = true;
		ui.btnPause->setText("Start");
	}
	else {
		configData->doPause = false;
		ui.btnPause->setText("Pause");
	}
}
