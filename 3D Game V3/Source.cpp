
#define _CRT_SECURE_NO_WARNINGS

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		 
#include <GL/freeglut.h>	
#endif


#include <string>
#include <vector>
#include <fstream>
#include <algorithm> 
#include <ctime>

unsigned int windowWidth = 800, windowHeight = 800;
unsigned char keyPressed[256];

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle)
{
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0)
	{
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message)
{
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK)
	{
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program)
{
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK)
	{
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}


// Regular shading and Environment Mapping
const char *vertexSource0 = R"(
#version 130 
precision highp float; 

in vec3 vertexPosition; 
in vec2 vertexTexCoord; 
in vec3 vertexNormal; 

uniform mat4 M, MInv, MVP;
uniform vec3 worldEye; 
uniform vec4 worldLightPosition; 

out vec2 texCoord; 
out vec3 worldNormal; 
out vec3 worldView; 
out vec3 worldLight; 

void main() { 
	texCoord = vertexTexCoord; 
	vec4 worldPosition =
		vec4(vertexPosition, 1) * M; 
	worldLight  =
		worldLightPosition.xyz * worldPosition.w - worldPosition.xyz * worldLightPosition.w; 
	worldView = worldEye - worldPosition.xyz; 
	worldNormal = (MInv * vec4(vertexNormal, 0.0)).xyz; 
	gl_Position = vec4(vertexPosition, 1) * MVP; 
	} 
)";

// Regular Shading
const char *fragmentSource0 = R"(
#version 130 
precision highp float; 

uniform sampler2D samplerUnit; 
uniform vec3 La, Le; 
uniform vec3 ka, kd, ks; 
uniform float shininess; 

in vec2 texCoord; 
in vec3 worldNormal; 
in vec3 worldView; 
in vec3 worldLight; 

out vec4 fragmentColor;

void main() { 
	vec3 N = normalize(worldNormal); 
	vec3 V = normalize(worldView); 
	vec3 L = normalize(worldLight); 
	vec3 H = normalize(V + N);
	vec3 texel = texture(samplerUnit, texCoord).xyz;
	vec3 color = 
		La * ka + 
		Le * kd * texel* max(0.0, dot(L, N)) + 
		Le * ks * pow(max(0.0, dot(H, N)), shininess); 
	fragmentColor = vec4(color, 1); 
	} 
)";

// Environment Mapping
const char *fragmentSource1 = R"(
#version 130 
precision highp float; 

uniform sampler2D samplerUnit; 
uniform vec3 La, Le; 
uniform vec3 ka, kd, ks; 
uniform float shininess; 

uniform samplerCube  environmentMap;

in vec2 texCoord; 
in vec3 worldNormal; 
in vec3 worldView; 
in vec3 worldLight; 

out vec4 fragmentColor;

void main() { 
	vec3 N = normalize(worldNormal); 
	vec3 V = normalize(worldView); 
	vec3 L = normalize(worldLight); 
	vec3 H = normalize(V + L);

	// for environment mapping
	vec3 R = 2 * N * dot(N, V) - V;
	vec3 texel1 = textureCube(environmentMap, R).xyz;
	vec3 texel2 = texture(samplerUnit, texCoord).xyz;

	vec3 texel = texel1 + texel2;

	vec3 color = 
		La * ka + 
		Le * kd * texel* max(0.0, dot(L, N)) + 
		Le * ks * pow(max(0.0, dot(H, N)), shininess); 
	fragmentColor = vec4(color, 1); 
	} 
)";

// Infinite Ground Plane
const char *vertexSource2 = R"(
#version 130 
precision highp float; 
	
in vec4 vertexPosition; 
in vec2 vertexTexCoord; 
in vec3 vertexNormal; 
uniform mat4 M, MInv, MVP; 
	
out vec2 texCoord; 
out vec4 worldPosition; 
out vec3 worldNormal; 
	
void main() { 
	texCoord = vertexTexCoord; 
	worldPosition = vertexPosition * M; 
	worldNormal = (MInv * vec4(vertexNormal, 0.0)).xyz; 
	gl_Position = vertexPosition * MVP; 
	} 
)";

// Infinte Ground Plane
const char *fragmentSource2 = R"(
#version 130
precision highp float;

uniform sampler2D samplerUnit;
uniform vec3 La, Le;
uniform vec3 ka, kd, ks;
uniform float shininess;
uniform vec3 worldEye;
uniform vec4 worldLightPosition;

in vec2 texCoord;
in vec4 worldPosition;
in vec3 worldNormal;

out vec4 fragmentColor;

void main() {
	vec3 N = normalize(worldNormal);
	vec3 V = normalize(worldEye * worldPosition.w - worldPosition.xyz);
	vec3 L = normalize(worldLightPosition.xyz * worldPosition.w - worldPosition.xyz * worldLightPosition.w);
	vec3 H = normalize(V + L);

	vec2 position = worldPosition.xy / worldPosition.w;
	vec2 tex = position.xy - floor(position.xy);
	vec3 texel = texture(samplerUnit, tex).xyz;
	vec3 color = La * ka + Le * kd * texel * max(0.0,dot(L, N)) + Le * ks * pow(max(0.0, dot(H, N)), shininess);

	fragmentColor = vec4(color, 1);
	}
)";

// Shadow Projection
const char *vertexSource3 = R"(
#version 130 
precision highp float; 

in vec3 vertexPosition; 
in vec2 vertexTexCoord; 
in vec3 vertexNormal; 

uniform mat4 M, MVP;
uniform vec4 worldLightPosition; 

void main() { 
	vec4 p = vec4(vertexPosition, 1) * M;
	
	vec3 s = vec3(p.x, p.y, p.z);
	s.y = -0.99;

	gl_Position = vec4(s, 1) * MVP; 
} 
)";

// Shadow Projection
const char *fragmentSource3 = R"(
#version 130 
precision highp float; 

uniform sampler2D samplerUnit; 
uniform samplerCube environmentMap;

out vec4 fragmentColor;

void main() { 
	fragmentColor = vec4(0, 0, 0, 1); 
} 
)";

// Rendering Environment
const char *vertexSource4 = R"(
#version 130 
precision highp float; 

in vec4 vertexPosition; 
uniform mat4 viewDirMatrix; 
	
out vec4 worldPosition; 
out vec3 viewDir; 
	
void main() 
{ 
	worldPosition = vertexPosition; 
	viewDir = (vertexPosition * viewDirMatrix).xyz; 
	gl_Position = vertexPosition; 
	gl_Position.z = 0.999999; 
} 
)";

// Rendering Environment
const char *fragmentSource4 = R"(
#version 130 
precision highp float; 

uniform sampler2D samplerUnit; 
uniform samplerCube  environmentMap; 
in vec4 worldPosition; 
in vec3 viewDir; 
out vec4 fragmentColor; 
	
void main() 
{ 
	vec3 texel = textureCube(environmentMap, viewDir).xyz; 
	fragmentColor = vec4(texel, 1); 
} 
)";

// Procedural Solid Texturing
const char *vertexSource5 = R"(
#version 130 
precision highp float; 

in vec3 vertexPosition; 
in vec2 vertexTexCoord; 
in vec3 vertexNormal; 

uniform mat4 M, MVP;

out vec4 worldPosition;

void main() { 
	worldPosition = vec4(vertexPosition, 1) * M; 
	gl_Position = vec4(vertexPosition, 1) * MVP; 
	} 
)";

// Procedural Solid Texturing
const char *fragmentSource5 = R"(
#version 130 
precision highp float; 

uniform sampler2D samplerUnit;

in vec4 worldPosition;

out vec4 fragmentColor;

float snoise(vec3 r) {
	vec3 s = vec3(7502, 22777, 4767); // random seed, kind of
	float f = 0.0;
	for (int i = 0; i<16; i++) {
		f += sin(dot(s, r) / 65536.0); // add a bunch of random sines
		s = mod(s, 32768.0) * 2.0 + floor(s / 32768.0);
		// generate next random
	}
	return f / 32.0 + 0.5; // result between 0 and 1
}

void main() { 
	float noise = snoise(worldPosition.xyz * 100.0);

	float w = worldPosition.x * 20.0 + pow(noise, 1.0) * 10.0;
	w = pow(sin(w)*0.5 + 0.5, 2.0);
	fragmentColor = vec4(mix(
		vec3(1, 1, 1),
		vec3(.5, .5, .5),
		w), 1);

	}
)";

// Procedural Solid Normal Mapping
const char *vertexSource6 = R"(
#version 130 
precision highp float; 

in vec3 vertexPosition; 
in vec2 vertexTexCoord; 
in vec3 vertexNormal; 

uniform mat4 M, MInv, MVP;
uniform vec3 worldEye; 
uniform vec4 worldLightPosition; 

out vec2 texCoord; 
out vec3 worldNormal; 
out vec3 worldView; 
out vec3 worldLight;
out vec4 worldPosition; 

void main() { 
	texCoord = vertexTexCoord; 
	worldPosition =
		vec4(vertexPosition, 1) * M; 
	worldLight  =
		worldLightPosition.xyz * worldPosition.w - worldPosition.xyz * worldLightPosition.w; 
	worldView = worldEye - worldPosition.xyz; 
	worldNormal = (MInv * vec4(vertexNormal, 0.0)).xyz; 
	gl_Position = vec4(vertexPosition, 1) * MVP; 
	} 
)";

// Procedural Solid Normal Mapping
const char *fragmentSource6 = R"(
#version 130 
precision highp float; 

uniform sampler2D samplerUnit; 
uniform vec3 La, Le; 
uniform vec3 ka, kd, ks; 
uniform float shininess; 

uniform samplerCube  environmentMap;

in vec2 texCoord; 
in vec3 worldNormal; 
in vec3 worldView; 
in vec3 worldLight; 
in vec4 worldPosition;

out vec4 fragmentColor;

vec3 snoiseGrad(vec3 r) {
  vec3 s = vec3(7502, 22777, 4767); // random seed, kind of
  vec3 f = vec3(0.0, 0.0, 0.0);
  for(int i=0; i<16; i++) {
    f += cos( dot(s, r) / 65536.0) * s; // add a bunch of random sines
    s = mod(s, 32768.0) * 2.0 + floor(s / 32768.0);
			// generate next random
  }
  return f / 65536.0;
}

void main() { 
	vec3 N = normalize(worldNormal) + snoiseGrad(worldPosition.xyz * 40.0) * 0.01; 
	vec3 V = normalize(worldView); 
	vec3 L = normalize(worldLight); 
	vec3 H = normalize(V + L);

	// for environment mapping
	vec3 R = 2 * N * dot(N, V) - V;
	vec3 texel1 = textureCube(environmentMap, R).xyz;

	vec3 color = 
		La * ka + 
		Le * kd * texel1* max(0.0, dot(L, N)) + 
		Le * ks * pow(max(0.0, dot(H, N)), shininess); 
	fragmentColor = vec4(color, 1); 
	} 
)";

struct vec2
{
	float x, y;

	vec2(float x = 0.0, float y = 0.0) : x(x), y(y) {}

	static vec2 random() { return vec2(((float)rand() / RAND_MAX) * 2 - 1, ((float)rand() / RAND_MAX) * 2 - 1); }

	vec2 operator+(const vec2& v) { return vec2(x + v.x, y + v.y); }

	vec2 operator-(const vec2& v) { return vec2(x - v.x, y - v.y); }

	vec2 operator*(float s) { return vec2(x * s, y * s); }

	vec2 operator/(float s) { return vec2(x / s, y / s); }

	float length() { return sqrt(x * x + y * y); }

	vec2 normalize() { return *this / length(); }
};

struct vec3
{
	float x, y, z;

	vec3(float x = 0.0, float y = 0.0, float z = 0.0) : x(x), y(y), z(z) {}

	static vec3 random() { return vec3(((float)rand() / RAND_MAX) * 2 - 1, ((float)rand() / RAND_MAX) * 2 - 1, ((float)rand() / RAND_MAX) * 2 - 1); }

	vec3 operator+(const vec3& v) { return vec3(x + v.x, y + v.y, z + v.z); }

	vec3 operator-(const vec3& v) { return vec3(x - v.x, y - v.y, z - v.z); }

	vec3 operator*(float s) { return vec3(x * s, y * s, z * s); }

	vec3 operator/(float s) { return vec3(x / s, y / s, z / s); }

	float length() { return sqrt(x * x + y * y + z * z); }

	vec3 normalize() { return *this / length(); }

	void print() { printf("%f \t %f \t %f \n", x, y, z); }
};

vec3 cross(const vec3& a, const vec3& b)
{
	return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}


// row-major matrix 4x4
struct mat4
{
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33)
	{
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right)
	{
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}

	operator float*() { return &m[0][0]; }
};

// 3D point in homogeneous coordinates
struct vec4
{
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1)
	{
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat)
	{
		vec4 result;
		for (int j = 0; j < 4; j++)
		{
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	vec4 operator+(const vec4& vec)
	{
		vec4 result(v[0] + vec.v[0], v[1] + vec.v[1], v[2] + vec.v[2], v[3] + vec.v[3]);
		return result;
	}
};

class   Mesh
{
	struct  Face
	{
		int       positionIndices[4];
		int       normalIndices[4];
		int       texcoordIndices[4];
		bool      isQuad;
	};

	std::vector<std::string*>	rows;
	std::vector<vec3*>		positions;
	std::vector<std::vector<Face*> >          submeshFaces;
	std::vector<vec3*>		normals;
	std::vector<vec2*>		texcoords;

	unsigned int vao;
	int nTriangles;

public:
	Mesh(const char *filename);
	~Mesh();

	void DrawModel();
};


Mesh::Mesh(const char *filename)
{
	std::fstream file(filename);
	if (!file.is_open())
	{
		return;
	}

	char buffer[256];
	while (!file.eof())
	{
		file.getline(buffer, 256);
		rows.push_back(new std::string(buffer));
	}

	submeshFaces.push_back(std::vector<Face*>());
	std::vector<Face*>* faces = &submeshFaces.at(submeshFaces.size() - 1);

	for (int i = 0; i < rows.size(); i++)
	{
		if (rows[i]->empty() || (*rows[i])[0] == '#')
			continue;
		else if ((*rows[i])[0] == 'v' && (*rows[i])[1] == ' ')
		{
			float tmpx, tmpy, tmpz;
			sscanf(rows[i]->c_str(), "v %f %f %f", &tmpx, &tmpy, &tmpz);
			positions.push_back(new vec3(tmpx, tmpy, tmpz));
		}
		else if ((*rows[i])[0] == 'v' && (*rows[i])[1] == 'n')
		{
			float tmpx, tmpy, tmpz;
			sscanf(rows[i]->c_str(), "vn %f %f %f", &tmpx, &tmpy, &tmpz);
			normals.push_back(new vec3(tmpx, tmpy, tmpz));
		}
		else if ((*rows[i])[0] == 'v' && (*rows[i])[1] == 't')
		{
			float tmpx, tmpy;
			sscanf(rows[i]->c_str(), "vt %f %f", &tmpx, &tmpy);
			texcoords.push_back(new vec2(tmpx, tmpy));
		}
		else if ((*rows[i])[0] == 'f')
		{
			if (count(rows[i]->begin(), rows[i]->end(), ' ') == 3)
			{
				Face* f = new Face();
				f->isQuad = false;
				sscanf(rows[i]->c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d",
					&f->positionIndices[0], &f->texcoordIndices[0], &f->normalIndices[0],
					&f->positionIndices[1], &f->texcoordIndices[1], &f->normalIndices[1],
					&f->positionIndices[2], &f->texcoordIndices[2], &f->normalIndices[2]);
				faces->push_back(f);
			}
			else
			{
				Face* f = new Face();
				f->isQuad = true;
				sscanf(rows[i]->c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d",
					&f->positionIndices[0], &f->texcoordIndices[0], &f->normalIndices[0],
					&f->positionIndices[1], &f->texcoordIndices[1], &f->normalIndices[1],
					&f->positionIndices[2], &f->texcoordIndices[2], &f->normalIndices[2],
					&f->positionIndices[3], &f->texcoordIndices[3], &f->normalIndices[3]);
				faces->push_back(f);
			}
		}
		else if ((*rows[i])[0] == 'g')
		{
			if (faces->size() > 0)
			{
				submeshFaces.push_back(std::vector<Face*>());
				faces = &submeshFaces.at(submeshFaces.size() - 1);
			}
		}
	}

	int numberOfTriangles = 0;
	for (int iSubmesh = 0; iSubmesh<submeshFaces.size(); iSubmesh++)
	{
		std::vector<Face*>& faces = submeshFaces.at(iSubmesh);

		for (int i = 0;i<faces.size();i++)
		{
			if (faces[i]->isQuad) numberOfTriangles += 2;
			else numberOfTriangles += 1;
		}
	}

	nTriangles = numberOfTriangles;

	float *vertexCoords = new float[numberOfTriangles * 9];
	float *vertexTexCoords = new float[numberOfTriangles * 6];
	float *vertexNormalCoords = new float[numberOfTriangles * 9];


	int triangleIndex = 0;
	for (int iSubmesh = 0; iSubmesh<submeshFaces.size(); iSubmesh++)
	{
		std::vector<Face*>& faces = submeshFaces.at(iSubmesh);

		for (int i = 0;i<faces.size();i++)
		{
			if (faces[i]->isQuad)
			{
				vertexTexCoords[triangleIndex * 6] = texcoords[faces[i]->texcoordIndices[0] - 1]->x;
				vertexTexCoords[triangleIndex * 6 + 1] = 1 - texcoords[faces[i]->texcoordIndices[0] - 1]->y;

				vertexTexCoords[triangleIndex * 6 + 2] = texcoords[faces[i]->texcoordIndices[1] - 1]->x;
				vertexTexCoords[triangleIndex * 6 + 3] = 1 - texcoords[faces[i]->texcoordIndices[1] - 1]->y;

				vertexTexCoords[triangleIndex * 6 + 4] = texcoords[faces[i]->texcoordIndices[2] - 1]->x;
				vertexTexCoords[triangleIndex * 6 + 5] = 1 - texcoords[faces[i]->texcoordIndices[2] - 1]->y;


				vertexCoords[triangleIndex * 9] = positions[faces[i]->positionIndices[0] - 1]->x;
				vertexCoords[triangleIndex * 9 + 1] = positions[faces[i]->positionIndices[0] - 1]->y;
				vertexCoords[triangleIndex * 9 + 2] = positions[faces[i]->positionIndices[0] - 1]->z;

				vertexCoords[triangleIndex * 9 + 3] = positions[faces[i]->positionIndices[1] - 1]->x;
				vertexCoords[triangleIndex * 9 + 4] = positions[faces[i]->positionIndices[1] - 1]->y;
				vertexCoords[triangleIndex * 9 + 5] = positions[faces[i]->positionIndices[1] - 1]->z;

				vertexCoords[triangleIndex * 9 + 6] = positions[faces[i]->positionIndices[2] - 1]->x;
				vertexCoords[triangleIndex * 9 + 7] = positions[faces[i]->positionIndices[2] - 1]->y;
				vertexCoords[triangleIndex * 9 + 8] = positions[faces[i]->positionIndices[2] - 1]->z;


				vertexNormalCoords[triangleIndex * 9] = normals[faces[i]->normalIndices[0] - 1]->x;
				vertexNormalCoords[triangleIndex * 9 + 1] = normals[faces[i]->normalIndices[0] - 1]->y;
				vertexNormalCoords[triangleIndex * 9 + 2] = normals[faces[i]->normalIndices[0] - 1]->z;

				vertexNormalCoords[triangleIndex * 9 + 3] = normals[faces[i]->normalIndices[1] - 1]->x;
				vertexNormalCoords[triangleIndex * 9 + 4] = normals[faces[i]->normalIndices[1] - 1]->y;
				vertexNormalCoords[triangleIndex * 9 + 5] = normals[faces[i]->normalIndices[1] - 1]->z;

				vertexNormalCoords[triangleIndex * 9 + 6] = normals[faces[i]->normalIndices[2] - 1]->x;
				vertexNormalCoords[triangleIndex * 9 + 7] = normals[faces[i]->normalIndices[2] - 1]->y;
				vertexNormalCoords[triangleIndex * 9 + 8] = normals[faces[i]->normalIndices[2] - 1]->z;

				triangleIndex++;


				vertexTexCoords[triangleIndex * 6] = texcoords[faces[i]->texcoordIndices[1] - 1]->x;
				vertexTexCoords[triangleIndex * 6 + 1] = 1 - texcoords[faces[i]->texcoordIndices[1] - 1]->y;

				vertexTexCoords[triangleIndex * 6 + 2] = texcoords[faces[i]->texcoordIndices[2] - 1]->x;
				vertexTexCoords[triangleIndex * 6 + 3] = 1 - texcoords[faces[i]->texcoordIndices[2] - 1]->y;

				vertexTexCoords[triangleIndex * 6 + 4] = texcoords[faces[i]->texcoordIndices[3] - 1]->x;
				vertexTexCoords[triangleIndex * 6 + 5] = 1 - texcoords[faces[i]->texcoordIndices[3] - 1]->y;


				vertexCoords[triangleIndex * 9] = positions[faces[i]->positionIndices[1] - 1]->x;
				vertexCoords[triangleIndex * 9 + 1] = positions[faces[i]->positionIndices[1] - 1]->y;
				vertexCoords[triangleIndex * 9 + 2] = positions[faces[i]->positionIndices[1] - 1]->z;

				vertexCoords[triangleIndex * 9 + 3] = positions[faces[i]->positionIndices[2] - 1]->x;
				vertexCoords[triangleIndex * 9 + 4] = positions[faces[i]->positionIndices[2] - 1]->y;
				vertexCoords[triangleIndex * 9 + 5] = positions[faces[i]->positionIndices[2] - 1]->z;

				vertexCoords[triangleIndex * 9 + 6] = positions[faces[i]->positionIndices[3] - 1]->x;
				vertexCoords[triangleIndex * 9 + 7] = positions[faces[i]->positionIndices[3] - 1]->y;
				vertexCoords[triangleIndex * 9 + 8] = positions[faces[i]->positionIndices[3] - 1]->z;


				vertexNormalCoords[triangleIndex * 9] = normals[faces[i]->normalIndices[1] - 1]->x;
				vertexNormalCoords[triangleIndex * 9 + 1] = normals[faces[i]->normalIndices[1] - 1]->y;
				vertexNormalCoords[triangleIndex * 9 + 2] = normals[faces[i]->normalIndices[1] - 1]->z;

				vertexNormalCoords[triangleIndex * 9 + 3] = normals[faces[i]->normalIndices[2] - 1]->x;
				vertexNormalCoords[triangleIndex * 9 + 4] = normals[faces[i]->normalIndices[2] - 1]->y;
				vertexNormalCoords[triangleIndex * 9 + 5] = normals[faces[i]->normalIndices[2] - 1]->z;

				vertexNormalCoords[triangleIndex * 9 + 6] = normals[faces[i]->normalIndices[3] - 1]->x;
				vertexNormalCoords[triangleIndex * 9 + 7] = normals[faces[i]->normalIndices[3] - 1]->y;
				vertexNormalCoords[triangleIndex * 9 + 8] = normals[faces[i]->normalIndices[3] - 1]->z;

				triangleIndex++;
			}
			else
			{
				vertexTexCoords[triangleIndex * 6] = texcoords[faces[i]->texcoordIndices[0] - 1]->x;
				vertexTexCoords[triangleIndex * 6 + 1] = 1 - texcoords[faces[i]->texcoordIndices[0] - 1]->y;

				vertexTexCoords[triangleIndex * 6 + 2] = texcoords[faces[i]->texcoordIndices[1] - 1]->x;
				vertexTexCoords[triangleIndex * 6 + 3] = 1 - texcoords[faces[i]->texcoordIndices[1] - 1]->y;

				vertexTexCoords[triangleIndex * 6 + 4] = texcoords[faces[i]->texcoordIndices[2] - 1]->x;
				vertexTexCoords[triangleIndex * 6 + 5] = 1 - texcoords[faces[i]->texcoordIndices[2] - 1]->y;

				vertexCoords[triangleIndex * 9] = positions[faces[i]->positionIndices[0] - 1]->x;
				vertexCoords[triangleIndex * 9 + 1] = positions[faces[i]->positionIndices[0] - 1]->y;
				vertexCoords[triangleIndex * 9 + 2] = positions[faces[i]->positionIndices[0] - 1]->z;

				vertexCoords[triangleIndex * 9 + 3] = positions[faces[i]->positionIndices[1] - 1]->x;
				vertexCoords[triangleIndex * 9 + 4] = positions[faces[i]->positionIndices[1] - 1]->y;
				vertexCoords[triangleIndex * 9 + 5] = positions[faces[i]->positionIndices[1] - 1]->z;

				vertexCoords[triangleIndex * 9 + 6] = positions[faces[i]->positionIndices[2] - 1]->x;
				vertexCoords[triangleIndex * 9 + 7] = positions[faces[i]->positionIndices[2] - 1]->y;
				vertexCoords[triangleIndex * 9 + 8] = positions[faces[i]->positionIndices[2] - 1]->z;


				vertexNormalCoords[triangleIndex * 9] = normals[faces[i]->normalIndices[0] - 1]->x;
				vertexNormalCoords[triangleIndex * 9 + 1] = normals[faces[i]->normalIndices[0] - 1]->y;
				vertexNormalCoords[triangleIndex * 9 + 2] = normals[faces[i]->normalIndices[0] - 1]->z;

				vertexNormalCoords[triangleIndex * 9 + 3] = normals[faces[i]->normalIndices[1] - 1]->x;
				vertexNormalCoords[triangleIndex * 9 + 4] = normals[faces[i]->normalIndices[1] - 1]->y;
				vertexNormalCoords[triangleIndex * 9 + 5] = normals[faces[i]->normalIndices[1] - 1]->z;

				vertexNormalCoords[triangleIndex * 9 + 6] = normals[faces[i]->normalIndices[2] - 1]->x;
				vertexNormalCoords[triangleIndex * 9 + 7] = normals[faces[i]->normalIndices[2] - 1]->y;
				vertexNormalCoords[triangleIndex * 9 + 8] = normals[faces[i]->normalIndices[2] - 1]->z;

				triangleIndex++;
			}
		}
	}

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	unsigned int vbo[3];
	glGenBuffers(3, &vbo[0]);

	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, nTriangles * 9 * sizeof(float), vertexCoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, nTriangles * 6 * sizeof(float), vertexTexCoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
	glBufferData(GL_ARRAY_BUFFER, nTriangles * 9 * sizeof(float), vertexNormalCoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	delete vertexCoords;
	delete vertexTexCoords;
	delete vertexNormalCoords;
}


void Mesh::DrawModel()
{
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, nTriangles * 3);
}


Mesh::~Mesh()
{
	for (unsigned int i = 0; i < rows.size(); i++) delete rows[i];
	for (unsigned int i = 0; i < positions.size(); i++) delete positions[i];
	for (unsigned int i = 0; i < submeshFaces.size(); i++)
		for (unsigned int j = 0; j < submeshFaces.at(i).size(); j++)
			delete submeshFaces.at(i).at(j);
	for (unsigned int i = 0; i < normals.size(); i++) delete normals[i];
	for (unsigned int i = 0; i < texcoords.size(); i++) delete texcoords[i];
}


class Material {

	vec3 ka, kd, ks;
	float shininess;

public:
	Material(vec3 ka, vec3 kd, vec3 ks, int shininess) : ka(ka), kd(kd), ks(ks), shininess(shininess) {}

	void SetMaterial(unsigned int shaderid) {
		int location = glGetUniformLocation(shaderid, "ka");
		if (location >= 0) glUniform3fv(location, 1, &ka.x);
		else printf("uniform ka cannot be set\n");

		location = glGetUniformLocation(shaderid, "kd");
		if (location >= 0) glUniform3fv(location, 1, &kd.x);
		else printf("uniform kd cannot be set\n");

		location = glGetUniformLocation(shaderid, "ks");
		if (location >= 0) glUniform3fv(location, 1, &ks.x);
		else printf("uniform ks cannot be set\n");

		location = glGetUniformLocation(shaderid, "shininess");
		if (location >= 0) glUniform1f(location, shininess);
		else printf("uniform shininess cannot be set\n");
	}

};

class Light {
	vec3 La, Le;
	vec4 worldLightPosition;

public:
	Light(vec3 La = vec3(1, 1, 1), vec3 Le = vec3(1, 1, 1), vec4 WLP = vec4(0, 2, 1, 0)) :
		La(La), Le(Le), worldLightPosition(WLP) {}

	void SetLight(unsigned int shaderid) {
		int location = glGetUniformLocation(shaderid, "La");
		if (location >= 0) glUniform3fv(location, 1, &La.x);
		else printf("uniform La cannot be set\n");

		location = glGetUniformLocation(shaderid, "Le");
		if (location >= 0) glUniform3fv(location, 1, &Le.x);
		else printf("uniform Le cannot be set\n");

		location = glGetUniformLocation(shaderid, "worldLightPosition");
		if (location >= 0) glUniform4fv(location, 1, &worldLightPosition.v[0]);
		else printf("uniform worldLightPosition cannot be set\n");
	}

	void SetPointLightSource(vec3& pos) {
		worldLightPosition = vec4(pos.x, pos.y, pos.z, 1);
	}

	void SetDirectionalLightSource(vec3& dir) {
		worldLightPosition = vec4(dir.x, dir.y, dir.z, 0);
	}
};

Light light;

class Camera {
	vec3  wEye, wLookat, wVup;
	float fov, asp, fp, bp;

	float velocity, angularVelocity;

public:
	Camera()
	{
		wEye = vec3(0.0, 0.0, 1.0);
		wLookat = vec3(0.0, 0.0, 0.0);
		wVup = vec3(0.0, 1.0, 0.0);
		fov = M_PI / 2.0; asp = 1.0; fp = 0.01; bp = 10.0;
		velocity = angularVelocity = 0.0;
	}

	void SetAspectRatio(float a) { asp = a; }

	vec3 getEyePosition() { return wEye; }

	void SetEyePosition(unsigned int shader) {

		int location = glGetUniformLocation(shader, "worldEye");
		if (location >= 0) glUniform3fv(location, 1, &wEye.x);
		else printf("uniform worldEye cannot be set\n");
	}

	mat4 GetViewMatrix() // view matrix 
	{
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);

		return
			mat4(
				1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f,
				-wEye.x, -wEye.y, -wEye.z, 1.0f) *
			mat4(
				u.x, v.x, w.x, 0.0f,
				u.y, v.y, w.y, 0.0f,
				u.z, v.z, w.z, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f);
	}

	mat4 GetProjectionMatrix() // projection matrix
	{
		float sy = 1 / tan(fov / 2);
		return mat4(
			sy / asp, 0.0f, 0.0f, 0.0f,
			0.0f, sy, 0.0f, 0.0f,
			0.0f, 0.0f, -(fp + bp) / (bp - fp), -1.0f,
			0.0f, 0.0f, -2 * fp*bp / (bp - fp), 0.0f);
	}

	mat4 GetInverseViewMatrix() {
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);
		return mat4(
			u.x, u.y, u.z, 0.0f,
			v.x, v.y, v.z, 0.0f,
			w.x, w.y, w.z, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);
	}
	mat4 GetInverseProjectionMatrix() {
		float sy = 1 / tan(fov / 2);
		return mat4(
			asp / sy, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0 / sy, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, (fp - bp) / 2.0 / fp / bp,
			0.0f, 0.0f, -1.0f, (fp + bp) / 2.0 / fp / bp);
	}


	void Move(float dt) {
		// update position of the camera
		//	Eye and Lookat must be changed/updated consistently

		vec3 velVec = (wLookat - wEye) * dt * velocity;

		wEye = wEye + velVec;
		wLookat = wLookat + velVec;

		vec3 aheadVec = wLookat - wEye;
		vec3 rightVec = cross(wVup * -1, aheadVec);

		// normalize
		vec3 rightNormal = rightVec.normalize();
		vec3 aheadNormal = aheadVec.normalize();
		//convert to radians
		float alphaRads = (angularVelocity*dt) / 180 * M_PI;
		// just change lookat
		// multiply with original length of ahead vector
		vec3 newNormalVec = aheadNormal * cos(alphaRads) + rightNormal * sin(alphaRads);
		wLookat = wEye + newNormalVec;

	}

	void Control() {

	}

	void UpdateCamera(vec3 position, vec3 ahead) {
		wLookat = vec3(position.x, 0.5, position.z);
		wEye = vec3(position.x + ahead.x, 0.5, position.z + ahead.z);
	}

};

extern "C" unsigned char* stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp);


Camera camera;

unsigned int shaderProgram0; // Regular Shading
unsigned int shaderProgram1; // Environment Mapping
unsigned int shaderProgram2; // Infinite Ground Plane
unsigned int shaderProgram3; // Shadow Projection
unsigned int shaderProgram4; // Rendering Environment
unsigned int shaderProgram5; // Procedural Solid Texturing
unsigned int shaderProgram6; // Procedural Solid Normal Mapping


							 // Textured Cube
class TextureCube
{
	unsigned int textureId;
public:
	TextureCube(
		const std::string& inputFileName0, const std::string& inputFileName1, const std::string& inputFileName2,
		const std::string& inputFileName3, const std::string& inputFileName4, const std::string& inputFileName5)
	{
		unsigned char* data[6]; int width[6]; int height[6]; int nComponents[6]; std::string filename[6];
		filename[0] = inputFileName0; filename[1] = inputFileName1; filename[2] = inputFileName2;
		filename[3] = inputFileName3; filename[4] = inputFileName4; filename[5] = inputFileName5;
		for (int i = 0; i < 6; i++) {
			data[i] = stbi_load(filename[i].c_str(), &width[i], &height[i], &nComponents[i], 0);
			if (data[i] == NULL) return;
		}
		glGenTextures(1, &textureId); glBindTexture(GL_TEXTURE_CUBE_MAP, textureId);
		for (int i = 0; i < 6; i++) {
			if (nComponents[i] == 4) glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA,
				width[i], height[i], 0, GL_RGBA, GL_UNSIGNED_BYTE, data[i]);
			if (nComponents[i] == 3) glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB,
				width[i], height[i], 0, GL_RGB, GL_UNSIGNED_BYTE, data[i]);
		}
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		for (int i = 0; i < 6; i++)  delete data[i];
	}

	void Bind(unsigned int shader)
	{
		int samplerCube = 1;
		int location = glGetUniformLocation(shader, "environmentMap");
		glUniform1i(location, samplerCube);
		glActiveTexture(GL_TEXTURE0 + samplerCube);
		glBindTexture(GL_TEXTURE_CUBE_MAP, textureId);
	}
};

TextureCube *environmentMap = 0;


class Object {
protected:
	vec3 position, scaling;
	float orientation;

	vec3 velocity;
	float angularVelocity;

	unsigned int shader;

	Material *material;

	bool alive;

public:
	Object(Material* mat, unsigned int sp) : material(mat), scaling(1.0, 1.0, 1.0), orientation(0.0),
		angularVelocity(0.0), shader(sp), alive(true) { }

	virtual void SetTransform()
	{
		mat4 scale(
			scaling.x, 0, 0, 0,
			0, scaling.y, 0, 0,
			0, 0, scaling.z, 0,
			0, 0, 0, 1);

		mat4 scaleInv(
			1.0 / scaling.x, 0, 0, 0,
			0, 1.0 / scaling.y, 0, 0,
			0, 0, 1.0 / scaling.z, 0,
			0, 0, 0, 1);

		float alpha = orientation / 180 * M_PI;

		mat4 rotate(
			cos(alpha), 0, sin(alpha), 0,
			0, 1, 0, 0,
			-sin(alpha), 0, cos(alpha), 0,
			0, 0, 0, 1);

		mat4 rotateInv(
			cos(alpha), 0, -sin(alpha), 0,
			0, 1, 0, 0,
			sin(alpha), 0, cos(alpha), 0,
			0, 0, 0, 1);

		mat4 translate(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			position.x, position.y, position.z, 1);

		mat4 translateInv(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-position.x, -position.y, -position.z, 1);

		mat4 M = scale * rotate * translate;

		mat4 MInv = translateInv * rotateInv * scaleInv;

		int location = glGetUniformLocation(shader, "MInv");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MInv);
		else printf("uniform MInv cannot be set\n");

		// Object modifies each object based on the camera's own position
		mat4 MVP = M * camera.GetViewMatrix() * camera.GetProjectionMatrix();

		location = glGetUniformLocation(shader, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVP);
		else printf("uniform MVP cannot be set\n");

		location = glGetUniformLocation(shader, "M");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, M);
		else printf("uniform M cannot be set\n");

	}

	void SetTransform2(unsigned int sp)
	{
		mat4 scale(
			scaling.x, 0, 0, 0,
			0, scaling.y, 0, 0,
			0, 0, scaling.z, 0,
			0, 0, 0, 1);

		float alpha = orientation / 180 * M_PI;

		mat4 rotate(
			cos(alpha), 0, sin(alpha), 0,
			0, 1, 0, 0,
			-sin(alpha), 0, cos(alpha), 0,
			0, 0, 0, 1);

		mat4 translate(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			position.x, position.y, position.z, 1);

		mat4 M = scale * rotate * translate;

		// Object modifies each object based on the camera's own position
		mat4 MVP = camera.GetViewMatrix() * camera.GetProjectionMatrix();

		int location = glGetUniformLocation(sp, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVP);
		else printf("uniform MVP cannot be set\n");

		location = glGetUniformLocation(sp, "M");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, M);
		else printf("uniform M cannot be set\n");
	}

	void Draw()
	{
		glUseProgram(shader);
		SetTransform();
		light.SetPointLightSource(camera.getEyePosition());
		camera.SetEyePosition(shader);
		light.SetLight(shader);
		material->SetMaterial(shader);
		DrawModel();

		if (environmentMap) environmentMap->Bind(shader);

	}

	void DrawShadow() {
		glUseProgram(shaderProgram3);
		SetTransform2(shaderProgram3);

		light.SetPointLightSource(camera.getEyePosition());
		light.SetLight(shaderProgram3);
		DrawModel();

	}

	virtual void DrawModel() = 0;

	virtual void Move(float dt)
	{
		position = position + velocity * dt;

		orientation = orientation + angularVelocity * dt;
	}

	virtual void Control()
	{

	}

	vec3 GetPosition() {
		return position;
	}

	bool isAlive() {
		return alive;
	}
};

Object *avatar = 0;

class Texture
{
	unsigned int textureId;

public:
	Texture(const std::string& inputFileName)
	{
		unsigned char* data;
		int width; int height; int nComponents = 4;

		data = stbi_load(inputFileName.c_str(), &width, &height, &nComponents, 0);

		if (data == NULL)
		{
			return;
		}

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);

		if (nComponents == 4) glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		if (nComponents == 3) glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		delete data;
	}

	void Bind(unsigned int shader)
	{
		int samplerUnit = 0;
		int location = glGetUniformLocation(shader, "samplerUnit");
		glUniform1i(location, samplerUnit);
		glActiveTexture(GL_TEXTURE0 + samplerUnit);
		glBindTexture(GL_TEXTURE_2D, textureId);
	}
};



class MeshInstance : public Object
{
	Texture *texture;
	Mesh * mesh;

public:

	MeshInstance(Texture* t, Mesh* m, Material* mat, unsigned int sp = shaderProgram0) : Object(mat, sp), texture(t), mesh(m)
	{
		scaling = vec3(0.05, 0.05, 0.05);
		position = vec3(0.0, -0.9, 0.0);
		angularVelocity = 20;
	}

	virtual void DrawModel()
	{
		texture->Bind(shader);

		glEnable(GL_DEPTH_TEST);
		mesh->DrawModel();
		glDisable(GL_DEPTH_TEST);
	}
};

class TexturedQuad : public Object {

	Texture *texture;
	unsigned int vao;

public:
	TexturedQuad(Texture* t, Material* mat, unsigned int sp = shaderProgram2) : Object(mat, sp), texture(t)
	{
		position = vec3(0, 0, 0);
		scaling = vec3(20, 1, 20);
		velocity = vec3(0, 0, 0);

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo[3];
		glGenBuffers(2, &vbo[0]);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		//static float vertexCoords[] = { 0, -1, 0, 1,    -0.5, 0, -0.5, 0,     0.5, 0, -0.5, 0,     
		//	0.5, 0, 0.5, 0,     -0.5, 0, 0.5, 0,     -0.5, 0, -0.5, 0, };
		static float vertexCoords[] = { 0, -1, 0, 1,    -1, 0, -1, 0,     1, 0, -1, 0,
			1, 0, 1, 0,     -1, 0, 1, 0,       -1, 0, -1, 0 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);


		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		//static float vertexTexCoord[] = { 0, 0,  1, 0,  1,1 ,  0, 1 };
		static float vertexTexCoord[] = { 0, 0,  0, 0,  0,0 ,  0, 0 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexTexCoord), vertexTexCoord, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
		static float vertexNormal[] = { 0,1,0,   0,1,0,    0,1,0,   0,1,0,   0,1,0,   0,1,0,  0,1,0 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexNormal), vertexNormal, GL_STATIC_DRAW);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	virtual void DrawModel()
	{
		texture->Bind(shader);

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 6);

		glDisable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
	}
};

// Fullscreen Quad for rendering the environment
class FullscreenQuad : public Object {

	Texture *texture;
	unsigned int vao;

public:
	FullscreenQuad(Texture* t, Material* mat, unsigned int sp = shaderProgram4) : Object(mat, sp), texture(t)
	{
		position = vec3(0, 0, 0);
		scaling = vec3(20, 1, 20);
		velocity = vec3(0, 0, 0);


		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo[3];
		glGenBuffers(2, &vbo[0]);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		static float vertexCoords[] = {
			-1.0, -1.0, 0.0, 1.0,
			1.0, -1.0, 0.0, 1.0,
			1.0, 1.0, 0.0, 1.0,
			-1.0, 1.0, 0.0, 1.0 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);


		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		static float vertexTexCoord[] = { 0, 0,  1, 0,  1,1 ,  0, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexTexCoord), vertexTexCoord, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
		static float vertexNormal[] = { 0,1,0,   0,1,0,    0,1,0,   0,1,0 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexNormal), vertexNormal, GL_STATIC_DRAW);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	virtual void DrawModel()
	{
		texture->Bind(shader);

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 6);

		glDisable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
	}

	void SetTransform() {
		mat4 viewDirMatrix =
			camera.GetInverseProjectionMatrix() * camera.GetInverseViewMatrix();

		int location = glGetUniformLocation(shader, "viewDirMatrix");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, viewDirMatrix);
		else printf("uniform viewDirMatrix cannot be set\n");
	}
};

class Tigger : public MeshInstance {
	float speed;

public:
	Tigger(Texture* t, Mesh* m, Material* mat, unsigned int sp = shaderProgram0) : MeshInstance(t, m, mat, sp)
	{
		speed = 0;
		position.y = -1;
	}

	void Move(float dt) {
		float ortnRads = orientation / 180 * M_PI;
		orientation = orientation + dt * angularVelocity;
		velocity = vec3(speed * cos(ortnRads), velocity.y, speed * sin(ortnRads));
		position = position + velocity*dt;
	}

	void Control() {
		if (keyPressed['j']) angularVelocity = -80;
		else if (keyPressed['l']) angularVelocity = 80;
		else angularVelocity = 0;

		if (keyPressed['i']) speed = -1;
		else if (keyPressed['k']) speed = 1;
		else speed = 0;

		// want to move camera when you move the tigger
		// take his ahead vector (moving direction)
		float ortnRads = orientation / 180 * M_PI;

		vec3 ahead = vec3(cos(ortnRads), velocity.y, sin(ortnRads));
		camera.UpdateCamera(position, ahead);
	}
};

class Seeker : public MeshInstance {

	float speed; // to tell it how fast to move/ if at all
	vec3 facing;


public:
	Seeker(Texture* t, Mesh* m, Material* mat, unsigned int sp = shaderProgram1) : MeshInstance(t, m, mat, sp)
	{
		position = vec3(1, -1, 0);
		scaling = vec3(0.03, 0.03, 0.03);
		speed = 0.0;
		angularVelocity = 0;

		facing = position - avatar->GetPosition();
	}

	void Move(float dt) {
		position = position + velocity * dt / 10;
	}

	void Control() {
		vec3 avatarPosition = avatar->GetPosition();
		velocity = vec3(avatarPosition.x - position.x, 0, avatarPosition.z - position.z);

		orientation = atan2(velocity.z, velocity.x) * 180 / M_PI + 180;
	}
};

class Tree : public MeshInstance {

public:
	Tree(vec3 position, vec3 scaling, Texture* t, Mesh* m, Material* mat, unsigned int sp = shaderProgram0) : MeshInstance(t, m, mat, sp)
	{
		this->position = position;
		velocity = 0;
		angularVelocity = 0;
		this->scaling = scaling;
	}
};

class Balloon : public MeshInstance {

public:
	Balloon(Texture *t, Mesh* m, Material* mat, unsigned int sp = shaderProgram1) : MeshInstance(t, m, mat, sp) {
		position = vec3(3, 3, 1);
		velocity = vec3(0, 20, 0);
		angularVelocity = 30;
	}

	void Move(float dt) {
		position.y = position.y + velocity.y * dt / 20;
		orientation = orientation + angularVelocity * dt;
	}

	void Control() {
		if (position.y > 4) {
			velocity.y = -20;
		}
		else if (position.y < 2) {
			velocity.y = 20;
		}
	}


};

class HeliBase : public MeshInstance {

public:

	HeliBase(Texture *t, Mesh* m, Material* mat, unsigned int sp = shaderProgram0) : MeshInstance(t, m, mat, sp) {
		position = vec3(-3, -1, 3);
		angularVelocity = 0;
		scaling = vec3(0.15, 0.15, 0.15);
	}
};

class HeliRotor : public MeshInstance {

public:

	HeliRotor(Texture *t, Mesh* m, Material* mat, unsigned int sp = shaderProgram0) : MeshInstance(t, m, mat, sp) {
		position = vec3(-3, 1.2, 3.5);
		angularVelocity = 300;
		scaling = vec3(0.1, 0.1, 0.1);
	}
};

// CheckPoint AI system
class CheckpointAvatar : public MeshInstance {
	vec3 limit1, limit2;

public:
	CheckpointAvatar(vec3 limit1, vec3 limit2, Texture *t, Mesh* m, Material* mat, unsigned int sp = shaderProgram0) : limit1(limit1), limit2(limit2),
		MeshInstance(t, m, mat, sp) {

		scaling = vec3(0.06, 0.03, 0.06);

		// start between the two points
		position = vec3((limit1.x + limit2.x) / 2, -1, (limit1.z + limit2.z) / 2);
		velocity = vec3(position.x - limit2.x, 0, position.z - limit2.z);
	}

	void Move(float dt) {
		position = position + velocity * dt / 10;
	}

	void Control() {
		// if too close to either limit, change direction
		if (vec3(abs(position.x - limit1.x), 0, abs(position.z - limit1.z)).length() < 0.1) {
			velocity = vec3(limit2.x - position.x, velocity.y, limit2.z - position.z);
		}
		else if (vec3(abs(position.x - limit2.x), 0, abs(position.z - limit2.z)).length() < 0.1) {
			velocity = vec3(limit1.x - position.x, velocity.y, limit1.z - position.z);
		}

		if (position.y >= -0.5) {
			velocity.y = -10;
		}
		else if (position.y <= -1)
			velocity.y = 10;

		orientation = atan2(velocity.z, velocity.x) * 180 / M_PI + 180;
	}
};

class Wheel : public MeshInstance {

public:
	Wheel(vec3 pos, Texture *t, Mesh* m, Material* mat, unsigned int sp = shaderProgram0) : MeshInstance(t, m, mat, sp) {
		position = pos;
		angularVelocity = 100;
	}

	void Control() {
		vec3 avPosition = avatar->GetPosition();

		// if too close to avatar, delete this object
		if (vec3(abs(position.x - avPosition.x), 0, abs(position.z - avPosition.z)).length() < 0.3) {
			alive = false;
		}
	}
};

class Chevy : public MeshInstance {

public:
	Chevy(Texture *t, Mesh* m, Material* mat, unsigned int sp = shaderProgram5) : MeshInstance(t, m, mat, sp) {
		position = vec3(-5, -0.6, -5);
		angularVelocity = 0;
		orientation = -90;
	}
};

class Scene
{
	std::vector<Texture*> textures;
	std::vector<Mesh*> meshes;
	std::vector<Object*> objects;
	std::vector<Material*> materials;

public:
	Scene()
	{

	}

	void Initialize()
	{
		environmentMap = new TextureCube("posx512.jpg", "negx512.jpg", "posy512.jpg", "negy512.jpg", "posz512.jpg", "negz512.jpg");
		//		environmentMap = new TextureCube("posx.jpg", "negx.jpg", "posy.jpg", "negy.jpg", "posz.jpg", "negz.jpg");

		textures.push_back(new Texture("tigger.png"));
		textures.push_back(new Texture("tree.png"));
		textures.push_back(new Texture("./balloon/balloon.png"));
		textures.push_back(new Texture("./heli/heli.png"));

		meshes.push_back(new Mesh("tigger.obj"));
		meshes.push_back(new Mesh("tree.obj"));
		meshes.push_back(new Mesh("./balloon/balloon.obj"));
		meshes.push_back(new Mesh("./heli/heli.obj"));
		meshes.push_back(new Mesh("./heli/mainrotor.obj"));
		meshes.push_back(new Mesh("./chevy/wheel.obj"));
		meshes.push_back(new Mesh("./chevy/chevy.obj"));

		// diffuse, no color effect
		materials.push_back(new Material(vec3(0.1, 0.1, 0.1), vec3(0.9, 0.9, 0.9), vec3(0, 0, 0), 0));
		// glossy, high shine, no color
		materials.push_back(new Material(vec3(0, 0, 0), vec3(0.9, 0.9, 0.9), vec3(0.3, 0.3, 0.3), 200));
		// glossy, high shine, light blue
		materials.push_back(new Material(vec3(0.1, 0.1, 0.1), vec3(0.212, 0.59, 0.85), vec3(0.3, 0.3, 0.3), 200));
		// diffuse, green hue
		materials.push_back(new Material(vec3(0.1, 0.1, 0.1), vec3(0, 0.9, 0), vec3(0, 0, 0), 0));

		// Sky
		objects.push_back(new FullscreenQuad(textures[0], materials[0]));

		// Ground Plane
		objects.push_back(new TexturedQuad(textures[1], materials[0]));

		objects.push_back(avatar = new Tigger(textures[0], meshes[0], materials[0]));
		objects.push_back(new Tree(vec3(1, -1, -4), vec3(.1, .1, .1), textures[1], meshes[1], materials[0]));

		objects.push_back(new Balloon(textures[2], meshes[2], materials[1]));
		objects.push_back(new Seeker(textures[0], meshes[0], materials[2]));

		objects.push_back(new HeliBase(textures[3], meshes[3], materials[0]));
		objects.push_back(new HeliRotor(textures[3], meshes[4], materials[0]));
		objects.push_back(new Chevy(textures[0], meshes[6], materials[0]));

		vec3 cpLimit1 = vec3(1, -1, 8);
		vec3 cpLimit2 = vec3(6, -1, 6);

		// Trees for CheckPoint AI
		objects.push_back(new Tree(cpLimit1, vec3(.1, .1, .1), textures[1], meshes[1], materials[0]));
		objects.push_back(new Tree(cpLimit2, vec3(.08, .08, .08), textures[1], meshes[1], materials[0]));

		objects.push_back(new CheckpointAvatar(cpLimit1, cpLimit2, textures[0], meshes[0], materials[1]));

		objects.push_back(new Wheel(vec3(1, 0, 1), textures[0], meshes[5], materials[0], shaderProgram6));
	}

	~Scene()
	{
		for (int i = 0; i < textures.size(); i++) delete textures[i];
		for (int i = 0; i < meshes.size(); i++) delete meshes[i];
		for (int i = 0; i < objects.size(); i++) delete objects[i];
		for (int i = 0; i < materials.size(); i++) delete materials[i];
		delete environmentMap;
		delete avatar;
	}

	void Draw()
	{
		for (int i = 2; i < objects.size(); i++) objects[i]->DrawShadow();
		for (int i = 0; i < objects.size(); i++) objects[i]->Draw();
	}

	void Move(float dt)
	{
		for (int i = 0; i < objects.size(); i++) objects[i]->Move(dt);
	}

	void Control()
	{
		for (int i = 0; i < objects.size(); i++) objects[i]->Control();

		std::vector<Object*> tmp = objects;
		objects.clear();
		for (int i = 0; i < tmp.size(); i++) {
			if (tmp[i]->isAlive())
				objects.push_back(tmp[i]);
			else {
				// push back a new wheel, it is the only thing that can 'die'
				objects.push_back(new Wheel(vec3(rand() % 11 - 5, 0, rand() % 11 - 5), textures[0], meshes[5], materials[0], shaderProgram6));
			}
		}

	}
};

Scene scene;

void createShaderProgram(const char *fragSrc, const char *vertSrc, unsigned int &sp) {
	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }

	glShaderSource(vertexShader, 1, &vertSrc, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }

	glShaderSource(fragmentShader, 1, &fragSrc, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	sp = glCreateProgram();
	if (!sp) { printf("Error in shader program creation\n"); exit(1); }

	glAttachShader(sp, vertexShader);
	glAttachShader(sp, fragmentShader);

	// Connect Attrib Arrays to input variables of the vertex shader
	glBindAttribLocation(sp, 0, "vertexPosition");	// vertexPosition gets values from Attrib Array 0
	glBindAttribLocation(sp, 1, "vertexTexCoord");  // vertexTexCoord gets values from Attrib Array 1
	glBindAttribLocation(sp, 2, "vertexNormal");  // vertexNormalCoord gets values from Attrib Array 2

												  // Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(sp, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

													// Program packaging
	glLinkProgram(sp);
	checkLinking(sp);
}


void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	srand(time(0));

	createShaderProgram(fragmentSource0, vertexSource0, shaderProgram0);
	createShaderProgram(fragmentSource1, vertexSource0, shaderProgram1);
	createShaderProgram(fragmentSource2, vertexSource2, shaderProgram2);
	createShaderProgram(fragmentSource3, vertexSource3, shaderProgram3);
	createShaderProgram(fragmentSource4, vertexSource4, shaderProgram4);
	createShaderProgram(fragmentSource5, vertexSource5, shaderProgram5);
	createShaderProgram(fragmentSource6, vertexSource6, shaderProgram6);

	scene.Initialize();

	for (int i = 0; i < 256; i++) keyPressed[i] = false;
}


void onKeyboard(unsigned char key, int x, int y)
{
	keyPressed[key] = true;
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int x, int y)
{
	keyPressed[key] = false;
	glutPostRedisplay();
}

void onExit() {
	glDeleteProgram(shaderProgram0);
	glDeleteProgram(shaderProgram1);
	glDeleteProgram(shaderProgram2);
	glDeleteProgram(shaderProgram3);
	glDeleteProgram(shaderProgram4);
	glDeleteProgram(shaderProgram5);
	glDeleteProgram(shaderProgram6);
	printf("exit");
}

void onDisplay() {

	glClearColor(0, 0.3, 0.5, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	scene.Draw();

	glutSwapBuffers();
}

void onReshape(int winWidth0, int winHeight0)
{
	glViewport(0, 0, winWidth0, winHeight0);

	windowWidth = winWidth0, windowHeight = winHeight0;

	camera.SetAspectRatio((float)windowWidth / windowHeight);
}


void onIdle() {
	double t = glutGet(GLUT_ELAPSED_TIME) * 0.001;
	static double lastTime = 0.0;
	double dt = t - lastTime;
	lastTime = t;

	scene.Control();
	scene.Move(dt);
	scene.Draw();

	camera.Control();
	camera.Move(dt);

	glutPostRedisplay();
}


int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(100, 100);
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow("3D Mesh");

#if !defined(__APPLE__)
	glewExperimental = true;
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);
	glutReshapeFunc(onReshape);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutIdleFunc(onIdle);

	glutMainLoop();
	onExit();
	return 1;
}
