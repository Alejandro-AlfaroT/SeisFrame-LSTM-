export const brainVertex = /* glsl */ `
attribute float aBrightness;
attribute float aRegion;
attribute float aSize;
uniform float uSelected;
uniform float uHover;
uniform float uPixelRatio;
uniform float uScale;
varying vec3 vColor;

void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float hasNormal = step(0.001, length(normal));
  vec3 n = hasNormal > 0.5 ? normalize(normalMatrix * normal) : vec3(0.0, 0.0, 1.0);
  float light = mix(0.85, 0.5 + 0.5 * max(dot(n, normalize(vec3(0.35, 0.6, 0.75))), 0.0), hasNormal);
  float rim = hasNormal * pow(1.0 - abs(n.z), 2.5) * 0.4;
  float sel = (uSelected >= 0.0 && abs(aRegion - uSelected) < 0.5) ? 1.0 : 0.0;
  float hov = (uHover >= 0.0 && abs(aRegion - uHover) < 0.5) ? 1.0 : 0.0;
  float dim = uSelected >= 0.0 ? mix(0.34, 1.25, sel) : 1.0;
  float b = 1.2 * aBrightness * (light + rim) * dim * (1.0 + 0.35 * hov);
  float depthFade = clamp(1.3 - (-mv.z - 1.6) * 0.3, 0.3, 1.0);
  vec3 white = vec3(0.9, 0.93, 0.97);
  vec3 lime = vec3(0.74, 1.0, 0.2);
  vColor = mix(white, lime, sel) * b * depthFade;
  gl_PointSize = aSize * uScale * uPixelRatio / -mv.z * (1.0 + 0.3 * sel);
  gl_Position = projectionMatrix * mv;
}
`;

export const nodeVertex = /* glsl */ `
attribute float aHighlight;
attribute float aRegion;
uniform float uSelected;
uniform float uAnyHighlight;
uniform float uPixelRatio;
uniform float uScale;
uniform float uTime;
varying vec3 vColor;

void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float inRegion = (uSelected >= 0.0 && abs(aRegion - uSelected) < 0.5) ? 1.0 : 0.0;
  float lit = max(aHighlight, inRegion);
  float dim = (uAnyHighlight > 0.5 || uSelected >= 0.0) ? mix(0.35, 1.0, lit) : 1.0;
  float pulse = 1.0 + 0.18 * aHighlight * sin(uTime * 3.0 + position.x * 20.0);
  vec3 white = vec3(1.0);
  vec3 lime = vec3(0.78, 1.0, 0.25);
  vColor = mix(white, lime, lit) * dim * 1.4;
  gl_PointSize = uScale * uPixelRatio / -mv.z * (1.0 + 0.9 * aHighlight) * pulse;
  gl_Position = projectionMatrix * mv;
}
`;

export const pointFragment = /* glsl */ `
varying vec3 vColor;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c);
  if (d > 0.5) discard;
  float a = smoothstep(0.5, 0.05, d);
  gl_FragColor = vec4(vColor * a, a);
}
`;
