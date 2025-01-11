#ifndef RENDERERINFO_H
#define RENDERERINFO_H

class rendererInfo {
public:
  Vec3 cameraDir;
  Vec3 cameraPos;
  Vec3 lightPos;

  RendererInfo();
  void copyToDevice();
};

#endif // RENDERERINFO_H
