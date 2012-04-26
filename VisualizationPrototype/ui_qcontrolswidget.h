/********************************************************************************
** Form generated from reading UI file 'qcontrolswidget.ui'
**
** Created: Wed Apr 25 16:13:09 2012
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QCONTROLSWIDGET_H
#define UI_QCONTROLSWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QControlsWidgetClass
{
public:
    QComboBox *comboBox;
    QLineEdit *radiusLineEdit;
    QLabel *radiusLabel;
    QLabel *isovalueLabel;
    QLabel *xLabel;
    QLineEdit *isovalueLineEdit;
    QLineEdit *xLineEdit;
    QLabel *objectLabel;
    QPushButton *applyPushButton;
    QLineEdit *yLineEdit;
    QLineEdit *zLineEdit;
    QLabel *yLabel;
    QLabel *zLabel;
    QGroupBox *modeSelectGroup;
    QRadioButton *normalModeRadioButton;
    QRadioButton *tensionModeRadioButton;
    QRadioButton *edgeModeRadioButton;

    void setupUi(QWidget *QControlsWidgetClass)
    {
        if (QControlsWidgetClass->objectName().isEmpty())
            QControlsWidgetClass->setObjectName(QString::fromUtf8("QControlsWidgetClass"));
        QControlsWidgetClass->resize(500, 378);
        comboBox = new QComboBox(QControlsWidgetClass);
        comboBox->setObjectName(QString::fromUtf8("comboBox"));
        comboBox->setGeometry(QRect(100, 30, 85, 27));
        radiusLineEdit = new QLineEdit(QControlsWidgetClass);
        radiusLineEdit->setObjectName(QString::fromUtf8("radiusLineEdit"));
        radiusLineEdit->setGeometry(QRect(150, 80, 113, 27));
        radiusLabel = new QLabel(QControlsWidgetClass);
        radiusLabel->setObjectName(QString::fromUtf8("radiusLabel"));
        radiusLabel->setGeometry(QRect(30, 80, 111, 17));
        isovalueLabel = new QLabel(QControlsWidgetClass);
        isovalueLabel->setObjectName(QString::fromUtf8("isovalueLabel"));
        isovalueLabel->setGeometry(QRect(40, 130, 67, 17));
        xLabel = new QLabel(QControlsWidgetClass);
        xLabel->setObjectName(QString::fromUtf8("xLabel"));
        xLabel->setGeometry(QRect(40, 170, 67, 17));
        isovalueLineEdit = new QLineEdit(QControlsWidgetClass);
        isovalueLineEdit->setObjectName(QString::fromUtf8("isovalueLineEdit"));
        isovalueLineEdit->setGeometry(QRect(130, 130, 113, 27));
        xLineEdit = new QLineEdit(QControlsWidgetClass);
        xLineEdit->setObjectName(QString::fromUtf8("xLineEdit"));
        xLineEdit->setGeometry(QRect(130, 170, 113, 27));
        objectLabel = new QLabel(QControlsWidgetClass);
        objectLabel->setObjectName(QString::fromUtf8("objectLabel"));
        objectLabel->setGeometry(QRect(30, 30, 67, 17));
        applyPushButton = new QPushButton(QControlsWidgetClass);
        applyPushButton->setObjectName(QString::fromUtf8("applyPushButton"));
        applyPushButton->setGeometry(QRect(210, 320, 97, 27));
        yLineEdit = new QLineEdit(QControlsWidgetClass);
        yLineEdit->setObjectName(QString::fromUtf8("yLineEdit"));
        yLineEdit->setGeometry(QRect(130, 210, 113, 27));
        zLineEdit = new QLineEdit(QControlsWidgetClass);
        zLineEdit->setObjectName(QString::fromUtf8("zLineEdit"));
        zLineEdit->setGeometry(QRect(130, 250, 113, 27));
        yLabel = new QLabel(QControlsWidgetClass);
        yLabel->setObjectName(QString::fromUtf8("yLabel"));
        yLabel->setGeometry(QRect(40, 210, 67, 17));
        zLabel = new QLabel(QControlsWidgetClass);
        zLabel->setObjectName(QString::fromUtf8("zLabel"));
        zLabel->setGeometry(QRect(40, 260, 67, 17));
        modeSelectGroup = new QGroupBox(QControlsWidgetClass);
        modeSelectGroup->setObjectName(QString::fromUtf8("modeSelectGroup"));
        modeSelectGroup->setGeometry(QRect(300, 50, 151, 131));
        normalModeRadioButton = new QRadioButton(modeSelectGroup);
        normalModeRadioButton->setObjectName(QString::fromUtf8("normalModeRadioButton"));
        normalModeRadioButton->setGeometry(QRect(10, 30, 114, 22));
        tensionModeRadioButton = new QRadioButton(modeSelectGroup);
        tensionModeRadioButton->setObjectName(QString::fromUtf8("tensionModeRadioButton"));
        tensionModeRadioButton->setGeometry(QRect(10, 60, 114, 22));
        edgeModeRadioButton = new QRadioButton(modeSelectGroup);
        edgeModeRadioButton->setObjectName(QString::fromUtf8("edgeModeRadioButton"));
        edgeModeRadioButton->setGeometry(QRect(10, 90, 114, 22));

        retranslateUi(QControlsWidgetClass);

        QMetaObject::connectSlotsByName(QControlsWidgetClass);
    } // setupUi

    void retranslateUi(QWidget *QControlsWidgetClass)
    {
        QControlsWidgetClass->setWindowTitle(QApplication::translate("QControlsWidgetClass", "QControlsWidget", 0, QApplication::UnicodeUTF8));
        radiusLabel->setText(QApplication::translate("QControlsWidgetClass", "Influence radius", 0, QApplication::UnicodeUTF8));
        isovalueLabel->setText(QApplication::translate("QControlsWidgetClass", "Isovalue", 0, QApplication::UnicodeUTF8));
        xLabel->setText(QApplication::translate("QControlsWidgetClass", "X", 0, QApplication::UnicodeUTF8));
        objectLabel->setText(QApplication::translate("QControlsWidgetClass", "Object", 0, QApplication::UnicodeUTF8));
        applyPushButton->setText(QApplication::translate("QControlsWidgetClass", "Apply", 0, QApplication::UnicodeUTF8));
        yLabel->setText(QApplication::translate("QControlsWidgetClass", "Y", 0, QApplication::UnicodeUTF8));
        zLabel->setText(QApplication::translate("QControlsWidgetClass", "Z", 0, QApplication::UnicodeUTF8));
        modeSelectGroup->setTitle(QApplication::translate("QControlsWidgetClass", "Display mode", 0, QApplication::UnicodeUTF8));
        normalModeRadioButton->setText(QApplication::translate("QControlsWidgetClass", "Normal", 0, QApplication::UnicodeUTF8));
        tensionModeRadioButton->setText(QApplication::translate("QControlsWidgetClass", "Tension", 0, QApplication::UnicodeUTF8));
        edgeModeRadioButton->setText(QApplication::translate("QControlsWidgetClass", "Edge", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class QControlsWidgetClass: public Ui_QControlsWidgetClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QCONTROLSWIDGET_H
