// requires "perpixellight.vs"

varying vec3 N;
varying vec3 L;
varying float density;

void main(void)
{
	const vec3 Color1 = vec3(0.2, 0.2, 0.4);
	const vec3 Color2 = vec3(0.7, 0.7, 0.7);

	vec3 l = normalize(L);
	vec3 n = normalize(N);
	vec3 H = normalize(l + vec3(0.0,0.0,1.0));

	// compute diffuse equation
	float NdotL = dot(n,l);
	vec4 diffuse = gl_Color * vec4(max(0.0,NdotL));

	float NdotH = max(0.0, dot(n,H));
	vec4 specular = vec4(0.0);
	const float specularExp = 128.0;
	if (NdotL > 0.0)
	  specular = vec4(pow(NdotH, specularExp));

	gl_FragColor = diffuse + specular;


	density = clamp(density, 0.0, 1.0);
	
	vec3 color = mix(Color1, Color2, density)*gl_FragColor.rgb;

	gl_FragColor.rgb = color;

	gl_FragColor.a = 1.0;
}
