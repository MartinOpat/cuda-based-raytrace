#ifndef SHADING_H
#define SHADING_H

#include "linalg/linalg.h"
#include "consts.h"

// TODO: Consider wrapping this in a class (?)
__device__ Vec3 phongShading(const Vec3& normal, const Vec3& lightDir, const Vec3& viewDir, const Vec3& baseColor) {
    Vec3 ambient = baseColor * ambientStrength;
    double diff = fmax(normal.dot(lightDir), 0.0);
    Vec3 diffuse = baseColor * (diffuseStrength * diff);

    Vec3 reflectDir = (normal * (2.0 * normal.dot(lightDir)) - lightDir).normalize();
    double spec = pow(fmax(viewDir.dot(reflectDir), 0.0), shininess);
    Vec3 specular = Vec3::init(1.0, 1.0, 1.0) * (specularStrength * spec);

    return ambient + diffuse + specular;
}

#endif // SHADING_H