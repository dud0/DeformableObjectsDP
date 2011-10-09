/**************************************************************************************/
/*                                                                                    */
/*  Visualization Library                                                             */
/*  http://www.visualizationlibrary.org                                               */
/*                                                                                    */
/*  Copyright (c) 2005-2010, Michele Bosi                                             */
/*  All rights reserved.                                                              */
/*                                                                                    */
/*  Redistribution and use in source and binary forms, with or without modification,  */
/*  are permitted provided that the following conditions are met:                     */
/*                                                                                    */
/*  - Redistributions of source code must retain the above copyright notice, this     */
/*  list of conditions and the following disclaimer.                                  */
/*                                                                                    */
/*  - Redistributions in binary form must reproduce the above copyright notice, this  */
/*  list of conditions and the following disclaimer in the documentation and/or       */
/*  other materials provided with the distribution.                                   */
/*                                                                                    */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   */
/*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     */
/*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE            */
/*  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR  */
/*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    */
/*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;      */
/*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON    */
/*  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS     */
/*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                      */
/*                                                                                    */
/**************************************************************************************/

#ifndef App_RotatingCube_INCLUDE_ONCE
#define App_RotatingCube_INCLUDE_ONCE

#include <vlGraphics/Applet.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlGraphics/SceneManagerActorTree.hpp>
#include <vlGraphics/Geometry.hpp>
#include <vlGraphics/Rendering.hpp>
#include <vlGraphics/Actor.hpp>
#include <vlCore/Time.hpp>
#include <vlGraphics/Effect.hpp>
#include <vlGraphics/Light.hpp>

class TetrahedronAnimator: public vl::ActorEventCallback
{
public:
	TetrahedronAnimator(vl::fvec3 a, vl::fvec3 b, vl::fvec3 c, vl::fvec3 d)
	:a(a),b(b),c(c),d(d)
	{
		vert3 = new vl::ArrayFloat3;

		vl::fvec3 verts[] = {
			a,c,b,
			a,b,d,
			b,c,d,
			d,c,a
		};
		vert3->resize(12);
		memcpy(vert3->ptr(), verts, sizeof(verts));

		mLastUpdate=0;
		grow=true;
	}

	virtual void onActorRenderStarted(vl::Actor*, vl::real frame_clock, const vl::Camera*, vl::Renderable* renderable, const vl::Shader*, int pass)
	{

		if (pass>0)
			return;

		const vl::real fps = 30.0f;

		if (frame_clock - mLastUpdate > 1.0f/fps)
		{
			mLastUpdate = frame_clock;

			vl::ref<vl::Geometry> geom = vl::cast<vl::Geometry>( renderable );

			if (grow)
			{
				a.x()-=0.1f;
				a.y()-=0.1f;
				a.z()-=0.1f;
				b.y()+=0.1f;
				c.z()+=0.1f;
				d.x()+=0.1f;
			} else {
				a.x()+=0.1f;
				a.y()+=0.1f;
				a.z()+=0.1f;
				b.y()-=0.1f;
				c.z()-=0.1f;
				d.x()-=0.1f;
			}


			if (b.y()>=30)
				grow=false;
			if(b.y()<=5)
				grow=true;

			vl::fvec3 verts[] = {
						a,c,b,
						a,b,d,
						b,c,d,
						d,c,a
					};
			memcpy(vert3->ptr(), verts, sizeof(verts));
			geom->setVertexArray(vert3.get());



			if (vl::Has_GL_ARB_vertex_buffer_object)
			{
				geom->vertexArray()->bufferObject()->setBufferData(vl::BU_DYNAMIC_DRAW, false);
			}

			geom->setBoundsDirty(true);
		}
	}

	virtual void onActorDelete(vl::Actor*) {}

protected:
	vl::real mLastUpdate;
	vl::fvec3 a,b,c,d;
	vl::ref<vl::ArrayFloat3> vert3;
	bool grow;
};

class TetraApplet: public vl::Applet
{
public:
  // called once after the OpenGL window has been opened 
  void initEvent()
  {
    // allocate the Transform 
    mTetraTransform = new vl::Transform;
    // bind the Transform with the transform tree of the rendring pipeline 
    rendering()->as<vl::Rendering>()->transform()->addChild( mTetraTransform.get() );

    a = new vl::fvec3(0,0,0);
	b = new vl::fvec3(0,5,0);
	c = new vl::fvec3(0,0,5);
	d = new vl::fvec3(5,0,0);
    // create the cube's Geometry and compute its normals to support lighting 
    cube = makeTetrahedron(*a,*b,*c,*d);
    cube->computeNormals();

    // setup the effect to be used to render the cube 
    vl::ref<vl::Effect> effect = new vl::Effect;
    // enable depth test and lighting 
    effect->shader()->enable(vl::EN_DEPTH_TEST);
    // add a Light to the scene, since no Transform is associated to the Light it will follow the camera 
    effect->shader()->setRenderState( new vl::Light, 0 );
    // enable the standard OpenGL lighting 
    effect->shader()->enable(vl::EN_LIGHTING);
    // set the front and back material color of the cube 
    // "gocMaterial" stands for "get-or-create Material"
    effect->shader()->gocMaterial()->setDiffuse( vl::green );

    // install our scene manager, we use the SceneManagerActorTree which is the most generic
    vl::ref<vl::SceneManagerActorTree> scene_manager = new vl::SceneManagerActorTree;
    rendering()->as<vl::Rendering>()->sceneManagers()->push_back(scene_manager.get());
    // add the cube to the scene using the previously defined effect and transform 
    vl::ref<vl::Actor> tetra_act=scene_manager->tree()->addActor( cube.get(), effect.get(), mTetraTransform.get()  );
    tetra_act->actorEventCallbacks()->push_back(new TetrahedronAnimator(*a,*b,*c,*d));
  }

  vl::ref<vl::Geometry> makeTetrahedron(vl::fvec3 a, vl::fvec3 b, vl::fvec3 c, vl::fvec3 d) {
  		vl::ref<vl::Geometry> geom = new vl::Geometry;
  		geom->setObjectName("Tetrahedron");

  		vl::ref<vl::ArrayFloat3> vert3 = new vl::ArrayFloat3;
  		geom->setVertexArray(vert3.get());

  		vl::fvec3 verts[] = {
  			a,c,b,
  			a,b,d,
  			b,c,d,
  			d,c,a
  		};

  		vl::ref<vl::DrawArrays> polys = new vl::DrawArrays(vl::PT_TRIANGLES, 0, 12);
  		geom->drawCalls()->push_back(polys.get());
  		vert3->resize(12);
  		memcpy(vert3->ptr(), verts, sizeof(verts));


  		return geom;
  	}

  // called every frame 
  virtual void updateScene()
  {
    // rotates the cube around the Y axis 45 degrees per second 
    vl::real degrees = vl::Time::currentTime() * 5.0f;
    vl::mat4 matrix = vl::mat4::getRotation( degrees, 0,1,0 );

    mTetraTransform->setLocalMatrix( matrix );
  }



protected:
  vl::ref<vl::Transform> mTetraTransform;
  vl::fvec3* a;
  		vl::fvec3* b;
  		vl::fvec3* c;
  		vl::fvec3* d;
  		vl::ref<vl::Geometry> cube;
};
// Have fun!

#endif
