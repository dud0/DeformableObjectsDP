varying vec3 N;
varying vec3 L;
varying float density;

void main(void)
{
	density = gl_Vertex.w;
	gl_Vertex.w = 1;
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	vec3 V = (gl_ModelViewMatrix * gl_Vertex).xyz;
	L = normalize(gl_LightSource[0].position.xyz - V);
	N = normalize(gl_NormalMatrix * gl_Normal);
	gl_FrontColor = gl_Color;
}
