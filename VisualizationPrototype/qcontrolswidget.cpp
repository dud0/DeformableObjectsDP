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
}

void QControlsWidget::saveObjectData() {
	configData->objectData[ui.comboBox->currentIndex()].radius=ui.radiusLineEdit->text().toFloat();
	configData->objectData[ui.comboBox->currentIndex()].isoValue=ui.isovalueLineEdit->text().toFloat();
	configData->objectData[ui.comboBox->currentIndex()].force[0]=ui.xLineEdit->text().toFloat();
	configData->objectData[ui.comboBox->currentIndex()].force[1]=ui.yLineEdit->text().toFloat();
	configData->objectData[ui.comboBox->currentIndex()].force[2]=ui.zLineEdit->text().toFloat();
}