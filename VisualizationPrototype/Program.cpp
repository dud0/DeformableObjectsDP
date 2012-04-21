#include "qcontrolswidget.h"
#include <vlCore/VisualizationLibrary.hpp>
#include <vlQt4/Qt4Widget.hpp>
#include "MCApplet.hpp"
#include "ConfigurationData.hpp"

using namespace vl;
using namespace vlQt4;

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);

  /* init Visualization Library */
  VisualizationLibrary::init();

  /* setup the OpenGL context format */
  OpenGLContextFormat format;
  //format.setDoubleBuffer(true);
  //format.setRGBABits( 8,8,8,8 );
  //format.setDepthBufferBits(24);
  //format.setStencilBufferBits(8);
  //format.setFullscreen(false);
  //format.setMultisampleSamples(16);
  //format.setMultisample(true);

  /* create the applet to be run */
  ConfigurationData *configData = new ConfigurationData;

  ref<MCApplet> applet = new MCApplet(configData);
  applet->initialize();
  /* create a native Qt4 window */
  ref<vlQt4::Qt4Widget> qt4_window = new vlQt4::Qt4Widget;
  /* bind the applet so it receives all the GUI events related to the OpenGLContext */
  qt4_window->addEventListener(applet.get());
  /* target the window so we can render on it */
  applet->rendering()->as<Rendering>()->renderer()->setFramebuffer( qt4_window->framebuffer() );
  /* black background */
  applet->rendering()->as<Rendering>()->camera()->viewport()->setClearColor( black );

  /* define the camera position and orientation */
  vec3 eye    = vec3(0,2,2); // camera position
  vec3 center = vec3(0,0,0);   // point the camera is looking at
  vec3 up     = vec3(0,1,0);   // up direction
  mat4 view_mat = mat4::getLookAt(eye, center, up);
  applet->rendering()->as<Rendering>()->camera()->setViewMatrix( view_mat );
  /* Initialize the OpenGL context and window properties */
  int x = 10;
  int y = 10;
  int width = 512;
  int height= 512;
  qt4_window->initQt4Widget( "Visualization Prototype", format, NULL, x, y, width, height );
  /* show the window */
  qt4_window->show();

  QControlsWidget *controlsWindow = new QControlsWidget(NULL, configData);

  //applet->setControlsWindow(controlsWindow);

  controlsWindow->setWindowTitle("Controls");
  controlsWindow->show();

  /* run the Win32 message loop */
  int val = app.exec();

  /* deallocate the window with all the OpenGL resources before shutting down Visualization Library */
  qt4_window = NULL;

  /* shutdown Visualization Library */
  VisualizationLibrary::shutdown();

  return val;
}
