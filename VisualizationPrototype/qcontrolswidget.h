#ifndef QCONTROLSWIDGET_H
#define QCONTROLSWIDGET_H

#include <QtGui/QWidget>
#include "ui_qcontrolswidget.h"
#include "ConfigurationData.hpp"

class QControlsWidget : public QWidget
{
    Q_OBJECT

public:
    QControlsWidget(QWidget *parent = 0, ConfigurationData * configData = 0);
    ~QControlsWidget();

private:
    Ui::QControlsWidgetClass ui;
    ConfigurationData * configData;

public slots:
	void showObjectData(int);
	void saveObjectData();
	void pause();
	void restart();
};

#endif // QCONTROLSWIDGET_H
