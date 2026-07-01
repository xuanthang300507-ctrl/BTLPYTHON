/*
 * ============================================================
 *  SMART PARKING SYSTEM  –  Web UI Server (C++) + Camera Capture
 *
 *  Yêu cầu: OpenCV đã cài đặt (vd: vcpkg install opencv hoặc
 *  apt install libopencv-dev trên Linux).
 *
 *  Build (Windows - MinGW, đã cài OpenCV qua vcpkg):
 *    g++ -std=c++17 -O2 parking_server.cpp -o parking_server.exe ^
 *        -I<path_to_opencv>/include ^
 *        -L<path_to_opencv>/lib ^
 *        -lopencv_world480 -lws2_32 -lpthread
 *
 *  Build (Linux):
 *    g++ -std=c++17 -O2 parking_server.cpp -o parking_server ^
 *        `pkg-config --cflags --libs opencv4` -lpthread
 *
 *  Chạy: ./parking_server
 *  Mở:   http://localhost:8080
 *
 *  Camera: mặc định mở webcam index 0 (CAMERA_INDEX). Để dùng
 *  camera IP/RTSP, đổi CAMERA_INDEX thành chuỗi URL, ví dụ:
 *    cv::VideoCapture cap("rtsp://user:pass@192.168.1.10/stream")
 *  Server sẽ liên tục ghi đè file current_frame.png, là file mà
 *  anpr.py (Python) đang theo dõi để chạy YOLO + EasyOCR.
 * ============================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <atomic>
#include <cctype>
// ── OpenCV (camera capture) ────────────────────────────────
#include <opencv2/opencv.hpp>
// ── Socket headers ─────────────────────────────────────────
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#define CLOSE(s) closesocket(s)
typedef int socklen_t;
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#define CLOSE(s) close(s)
#endif

// ============================================================
//  CAMERA CAPTURE  (cầu nối ảnh cho Python AI Engine)
//  Liên tục chụp khung hình từ camera (USB/webcam/IP cam) và
//  ghi đè vào file CURRENT_FRAME_FILE, file mà anpr.py (Python)
//  đang theo dõi (giám sát mtime) để chạy YOLO + OCR.
// ============================================================

#define CAMERA_INDEX 0 // 0 = webcam mặc định. Đổi thành chuỗi
                       // "rtsp://..." hoặc "http://ip:port/video"
                       // nếu dùng camera IP (sửa kiểu khai báo bên dưới)
#define CURRENT_FRAME_FILE "current_frame.bmp"
// vì bản OpenCV (MSYS2 pacman) có thể thiếu
// module ghi JPEG, từng gây crash imwrite()
#define SCAN_TRIGGER_FILE "scan_trigger.flag" // file cờ hiệu: web bấm nút → ghi file này
                                              // → anpr.py (Python) phát hiện và mới chạy YOLO+OCR
#define CAPTURE_INTERVAL_MS 500               // chu kỳ chụp ảnh (ms)

std::atomic<bool> g_cameraRunning{true};
std::atomic<bool> g_cameraOK{false};

void cameraCaptureLoop()
{
  cv::VideoCapture cap(CAMERA_INDEX, cv::CAP_DSHOW);

  if (!cap.isOpened())
  {
    std::cerr << "⚠️  [CAMERA] Không thể mở camera index " << CAMERA_INDEX
              << ". Luồng chụp ảnh sẽ thử kết nối lại...\n";
  }

  cv::Mat frame;
  while (g_cameraRunning)
  {
    if (!cap.isOpened())
    {
      cap.open(CAMERA_INDEX, cv::CAP_DSHOW);
      if (!cap.isOpened())
      {
        g_cameraOK = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        continue;
      }
      std::cout << "📷 [CAMERA] Đã kết nối camera index " << CAMERA_INDEX << "\n";
    }

    try
    {
      if (cap.read(frame) && !frame.empty())
      {
        g_cameraOK = true;
        if (!cv::imwrite(CURRENT_FRAME_FILE, frame))
        {
          g_cameraOK = false;
        }
      }
      else
      {
        g_cameraOK = false;
      }
    }
    catch (const cv::Exception &e)
    {
      static bool warned = false;
      if (!warned)
      {
        std::cerr << "⚠️  [CAMERA] Lỗi ghi ảnh (imwrite): " << e.what()
                  << " — kiểm tra OpenCV có module imgcodecs đầy đủ không.\n";
        warned = true;
      }
      g_cameraOK = false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(CAPTURE_INTERVAL_MS));
  }

  if (cap.isOpened())
  {
    cap.release();
  }
}

// ============================================================
//  DATA STRUCTURES
// ============================================================

#define ROW 5
#define COL 8
#define MAX_SLOTS 20
#define MAX_CARS 100
#define MAX_HISTORY 500

struct Position
{
  int x, y;
};

struct Node
{
  Position pos;
  int g, h, f;
  bool visited;
  Position parent;
};

struct Slot
{
  char id[10];
  Position pos;
  bool occupied;
};

struct Car
{
  char plate[20];
  char slotID[10];
  char checkIn[32]; // timestamp
};

struct HistoryEntry
{
  char text[120];
};

struct ParkingLot
{
  Slot slots[MAX_SLOTS];
  Car cars[MAX_CARS];
  HistoryEntry history[MAX_HISTORY];
  Node nodes[ROW][COL];
  int slotCount, carCount, historyCount;
};

// ============================================================
//  MAP  (0=road, 1=wall, 2=parking slot)
// ============================================================

int parkingMap[ROW][COL] = {
    {0, 0, 0, 0, 2, 0, 2, 0},
    {0, 1, 1, 0, 2, 0, 2, 0},
    {2, 0, 0, 0, 2, 0, 2, 0},
    {2, 1, 1, 0, 2, 0, 2, 0},
    {2, 2, 2, 2, 2, 2, 2, 2}};

// ============================================================
//  A* PATH-FINDER
// ============================================================

struct QueueNode
{
  Node *ptr;
  int priority;
};

bool isValid(int x, int y)
{
  return x >= 0 && x < ROW && y >= 0 && y < COL && parkingMap[x][y] != 1;
}

int heuristic(Position a, Position b)
{
  return abs(a.x - b.x) + abs(a.y - b.y);
}

void resetNodes(Node nodes[ROW][COL])
{
  for (int i = 0; i < ROW; i++)
    for (int j = 0; j < COL; j++)
    {
      nodes[i][j].g = 999999;
      nodes[i][j].h = 0;
      nodes[i][j].f = 999999;
      nodes[i][j].visited = false;
      nodes[i][j].parent.x = -1;
      nodes[i][j].parent.y = -1;
    }
}

void insertPQ(QueueNode q[], int &sz, Node *n, int pri)
{
  if (sz >= ROW * COL)
    return;
  int pos = sz;
  while (pos > 0 && q[pos - 1].priority > pri)
  {
    q[pos] = q[pos - 1];
    pos--;
  }
  q[pos].ptr = n;
  q[pos].priority = pri;
  sz++;
}

QueueNode extractMin(QueueNode q[], int &sz)
{
  QueueNode r = q[0];
  for (int i = 0; i < sz - 1; i++)
    q[i] = q[i + 1];
  sz--;
  return r;
}

int getDistance(Node nodes[ROW][COL], Position s, Position g)
{
  QueueNode open[ROW * COL];
  int sz = 0;
  resetNodes(nodes);
  Node *sn = &nodes[s.x][s.y];
  sn->pos = s;
  sn->g = 0;
  sn->h = heuristic(s, g);
  sn->f = sn->h;
  insertPQ(open, sz, sn, sn->f);
  int dx[] = {-1, 1, 0, 0}, dy[] = {0, 0, -1, 1};
  while (sz > 0)
  {
    QueueNode cur = extractMin(open, sz);
    if (cur.ptr->visited)
      continue;
    cur.ptr->visited = true;
    if (cur.ptr->pos.x == g.x && cur.ptr->pos.y == g.y)
      return cur.ptr->g;
    for (int i = 0; i < 4; i++)
    {
      int nx = cur.ptr->pos.x + dx[i], ny = cur.ptr->pos.y + dy[i];
      if (!isValid(nx, ny) || nodes[nx][ny].visited)
        continue;
      Node *nb = &nodes[nx][ny];
      int ng = cur.ptr->g + 1;
      if (ng < nb->g)
      {
        nb->pos.x = nx;
        nb->pos.y = ny;
        nb->g = ng;
        nb->h = heuristic({nx, ny}, g);
        nb->f = nb->g + nb->h;
        nb->parent = cur.ptr->pos;
        insertPQ(open, sz, nb, nb->f);
      }
    }
  }
  return 999999;
}

// reconstruct path as JSON array
std::string reconstructPath(Node nodes[ROW][COL], int ex, int ey)
{
  if (ex == -1 || ey == -1)
    return "[]";
  std::vector<std::pair<int, int>> path;
  int cx = ex, cy = ey;
  while (cx != -1 && cy != -1)
  {
    path.push_back({cx, cy});
    int px = nodes[cx][cy].parent.x, py = nodes[cx][cy].parent.y;
    cx = px;
    cy = py;
  }
  std::reverse(path.begin(), path.end());
  std::string s = "[";
  for (size_t i = 0; i < path.size(); i++)
  {
    if (i)
      s += ",";
    s += "[" + std::to_string(path[i].first) + "," + std::to_string(path[i].second) + "]";
  }
  s += "]";
  return s;
}

std::string findPathJSON(Node nodes[ROW][COL], Position s, Position g)
{
  QueueNode open[ROW * COL];
  int sz = 0;
  resetNodes(nodes);
  Node *sn = &nodes[s.x][s.y];
  sn->pos = s;
  sn->g = 0;
  sn->h = heuristic(s, g);
  sn->f = sn->h;
  insertPQ(open, sz, sn, sn->f);
  int dx[] = {-1, 1, 0, 0}, dy[] = {0, 0, -1, 1};
  while (sz > 0)
  {
    QueueNode cur = extractMin(open, sz);
    if (cur.ptr->visited)
      continue;
    cur.ptr->visited = true;
    if (cur.ptr->pos.x == g.x && cur.ptr->pos.y == g.y)
      return reconstructPath(nodes, g.x, g.y);
    for (int i = 0; i < 4; i++)
    {
      int nx = cur.ptr->pos.x + dx[i], ny = cur.ptr->pos.y + dy[i];
      if (!isValid(nx, ny) || nodes[nx][ny].visited)
        continue;
      Node *nb = &nodes[nx][ny];
      int ng = cur.ptr->g + 1;
      if (ng < nb->g)
      {
        nb->pos.x = nx;
        nb->pos.y = ny;
        nb->g = ng;
        nb->h = heuristic({nx, ny}, g);
        nb->f = nb->g + nb->h;
        nb->parent = cur.ptr->pos;
        insertPQ(open, sz, nb, nb->f);
      }
    }
  }
  return "[]";
}

// ============================================================
//  PARKING LOT LOGIC
// ============================================================

ParkingLot lot;

void initLot()
{
  lot.slotCount = 0;
  lot.carCount = 0;
  lot.historyCount = 0;
  int num = 1;
  for (int i = 0; i < ROW; i++)
    for (int j = 0; j < COL; j++)
      if (parkingMap[i][j] == 2 && lot.slotCount < MAX_SLOTS)
      {
        sprintf(lot.slots[lot.slotCount].id, "P%d", num++);
        lot.slots[lot.slotCount].pos.x = i;
        lot.slots[lot.slotCount].pos.y = j;
        lot.slots[lot.slotCount].occupied = false;
        lot.slotCount++;
      }
}

int findCar(const char *plate)
{
  for (int i = 0; i < lot.carCount; i++)
    if (strcmp(lot.cars[i].plate, plate) == 0)
      return i;
  return -1;
}

int findSlot(const char *id)
{
  for (int i = 0; i < lot.slotCount; i++)
    if (strcmp(lot.slots[i].id, id) == 0)
      return i;
  return -1;
}

int findNearestSlot()
{
  Position start = {0, 0};
  int best = -1, bestDist = 999999;
  for (int i = 0; i < lot.slotCount; i++)
  {
    if (!lot.slots[i].occupied)
    {
      int d = getDistance(lot.nodes, start, lot.slots[i].pos);
      if (d < bestDist)
      {
        bestDist = d;
        best = i;
      }
    }
  }
  return best;
}

void addHistory(const char *text)
{
  if (lot.historyCount < MAX_HISTORY)
    strcpy(lot.history[lot.historyCount++].text, text);
}

void saveData()
{
  std::ofstream f("parking_data.txt");
  for (int i = 0; i < lot.carCount; i++)
    f << lot.cars[i].plate << " " << lot.cars[i].slotID << " " << lot.cars[i].checkIn << "\n";
}

void loadData()
{
  std::ifstream f("parking_data.txt");
  if (!f)
    return;
  std::string line;
  while (std::getline(f, line) && lot.carCount < MAX_CARS)
  {
    if (line.empty())
      continue;

    std::istringstream ss(line);
    std::string plate, slotID, ts;
    if (!(ss >> plate >> slotID))
      continue; // dòng hỏng/không đủ trường, bỏ qua

    // Phần còn lại của dòng (sau slotID) chính là checkIn timestamp,
    // có thể chứa khoảng trắng (vd "08:51 03/06/2026") nên phải lấy
    // nguyên phần còn lại thay vì đọc theo từng "từ" như trước
    // (lỗi cũ khiến ngày tháng bị cắt rời và lệch sang dòng kế tiếp).
    std::getline(ss, ts);
    size_t start = ts.find_first_not_of(' ');
    ts = (start == std::string::npos) ? "" : ts.substr(start);

    strncpy(lot.cars[lot.carCount].plate, plate.c_str(), sizeof(lot.cars[lot.carCount].plate) - 1);
    lot.cars[lot.carCount].plate[sizeof(lot.cars[lot.carCount].plate) - 1] = '\0';
    strncpy(lot.cars[lot.carCount].slotID, slotID.c_str(), sizeof(lot.cars[lot.carCount].slotID) - 1);
    lot.cars[lot.carCount].slotID[sizeof(lot.cars[lot.carCount].slotID) - 1] = '\0';
    strncpy(lot.cars[lot.carCount].checkIn, ts.c_str(), sizeof(lot.cars[lot.carCount].checkIn) - 1);
    lot.cars[lot.carCount].checkIn[sizeof(lot.cars[lot.carCount].checkIn) - 1] = '\0';

    int si = findSlot(slotID.c_str());
    if (si != -1)
      lot.slots[si].occupied = true;
    lot.carCount++;
  }
}

void loadHistory()
{
  std::ifstream f("history.txt");
  if (!f)
    return;
  char line[120];
  while (f.getline(line, sizeof(line)) && lot.historyCount < MAX_HISTORY)
    strcpy(lot.history[lot.historyCount++].text, line);
}

void saveHistory(const char *text)
{
  std::ofstream f("history.txt", std::ios::app);
  f << text << "\n";
}

// ============================================================
//  URL DECODE
// ============================================================

std::string urlDecode(const std::string &s)
{
  std::string r;
  for (size_t i = 0; i < s.size(); i++)
  {
    if (s[i] == '%' && i + 2 < s.size())
    {
      int v;
      sscanf(s.substr(i + 1, 2).c_str(), "%x", &v);
      r += (char)v;
      i += 2;
    }
    else if (s[i] == '+')
      r += ' ';
    else
      r += s[i];
  }
  return r;
}

std::string getParam(const std::string &body, const std::string &key)
{
  size_t p = body.find(key + "=");
  if (p == std::string::npos)
    return "";
  p += key.size() + 1;
  size_t e = body.find('&', p);
  return urlDecode(e == std::string::npos ? body.substr(p) : body.substr(p, e - p));
}

// ============================================================
//  JSON BUILDERS
// ============================================================

std::string lotStatusJSON()
{
  int occupied = 0;
  for (int i = 0; i < lot.slotCount; i++)
    if (lot.slots[i].occupied)
      occupied++;
  int free = lot.slotCount - occupied;

  std::string s = "{";
  s += "\"total\":" + std::to_string(lot.slotCount) + ",";
  s += "\"occupied\":" + std::to_string(occupied) + ",";
  s += "\"free\":" + std::to_string(free) + ",";
  s += "\"camera\":" + std::string(g_cameraOK ? "true" : "false") + ",";
  s += "\"cars\":[";
  for (int i = 0; i < lot.carCount; i++)
  {
    if (i)
      s += ",";
    s += "{\"plate\":\"" + std::string(lot.cars[i].plate) + "\",";
    s += "\"slot\":\"" + std::string(lot.cars[i].slotID) + "\",";
    s += "\"checkin\":\"" + std::string(lot.cars[i].checkIn) + "\"}";
  }
  s += "],\"slots\":[";
  for (int i = 0; i < lot.slotCount; i++)
  {
    if (i)
      s += ",";
    s += "{\"id\":\"" + std::string(lot.slots[i].id) + "\",";
    s += "\"x\":" + std::to_string(lot.slots[i].pos.x) + ",";
    s += "\"y\":" + std::to_string(lot.slots[i].pos.y) + ",";
    s += "\"occupied\":" + (lot.slots[i].occupied ? std::string("true") : std::string("false")) + "}";
  }
  s += "],\"history\":[";
  int start = std::max(0, lot.historyCount - 20);
  for (int i = lot.historyCount - 1; i >= start; i--)
  {
    if (i != lot.historyCount - 1)
      s += ",";
    // escape quotes
    std::string txt = lot.history[i].text;
    std::string esc;
    for (char c : txt)
    {
      if (c == '"')
        esc += "\\\"";
      else
        esc += c;
    }
    s += "\"" + esc + "\"";
  }
  s += "]}";
  return s;
}

// ============================================================
//  ACTION HANDLERS  → return JSON {ok, msg, path, steps}
// ============================================================

std::string handlePark(const std::string &body)
{
  std::string plate = getParam(body, "plate");
  if (plate.empty())
    return "{\"ok\":false,\"msg\":\"Thiếu biển số xe\"}";
  // sanitize
  for (char c : plate)
    if (!isalnum(c) && c != '-')
      return "{\"ok\":false,\"msg\":\"Biển số không hợp lệ\"}";

  if (findCar(plate.c_str()) != -1)
    return "{\"ok\":false,\"msg\":\"Xe đã có trong bãi!\"}";
  if (lot.carCount >= MAX_CARS)
    return "{\"ok\":false,\"msg\":\"Hệ thống đã đầy!\"}";

  int si = findNearestSlot();
  if (si == -1)
    return "{\"ok\":false,\"msg\":\"Bãi đã đầy!\"}";

  // A* path
  Position start = {0, 0}, goal = lot.slots[si].pos;
  std::string pathArr = findPathJSON(lot.nodes, start, goal);
  int dist = getDistance(lot.nodes, start, goal);

  if (plate.size() >= sizeof(lot.cars[lot.carCount].plate))
    return "{\"ok\":false,\"msg\":\"Biển số quá dài!\"}";

  lot.slots[si].occupied = true;
  strncpy(lot.cars[lot.carCount].plate, plate.c_str(), sizeof(lot.cars[lot.carCount].plate) - 1);
  lot.cars[lot.carCount].plate[sizeof(lot.cars[lot.carCount].plate) - 1] = '\0';
  strncpy(lot.cars[lot.carCount].slotID, lot.slots[si].id, sizeof(lot.cars[lot.carCount].slotID) - 1);
  lot.cars[lot.carCount].slotID[sizeof(lot.cars[lot.carCount].slotID) - 1] = '\0';
  time_t t = time(0);
  struct tm *tm = localtime(&t);
  char ts[32];
  strftime(ts, sizeof(ts), "%H:%M %d/%m/%Y", tm);
  strcpy(lot.cars[lot.carCount].checkIn, ts);
  lot.carCount++;

  char hist[120];
  sprintf(hist, "[IN ] %s → %s  (%s)", plate.c_str(), lot.slots[si].id, ts);
  addHistory(hist);
  saveHistory(hist);
  saveData();

  return "{\"ok\":true,\"msg\":\"Gửi xe thành công!\","
         "\"slot\":\"" +
         std::string(lot.slots[si].id) + "\","
                                         "\"x\":" +
         std::to_string(lot.slots[si].pos.x) + ","
                                               "\"y\":" +
         std::to_string(lot.slots[si].pos.y) + ","
                                               "\"steps\":" +
         std::to_string(dist) + ","
                                "\"path\":" +
         pathArr + "}";
}

std::string handleLeave(const std::string &body)
{
  std::string plate = getParam(body, "plate");
  if (plate.empty())
    return "{\"ok\":false,\"msg\":\"Thiếu biển số xe\"}";

  int idx = findCar(plate.c_str());
  if (idx == -1)
    return "{\"ok\":false,\"msg\":\"Không tìm thấy xe trong bãi!\"}";

  char slotID[10];
  strcpy(slotID, lot.cars[idx].slotID);
  int si = findSlot(slotID);
  if (si != -1)
    lot.slots[si].occupied = false;

  for (int i = idx; i < lot.carCount - 1; i++)
    lot.cars[i] = lot.cars[i + 1];
  lot.carCount--;

  time_t t = time(0);
  struct tm *tm = localtime(&t);
  char ts[32];
  strftime(ts, sizeof(ts), "%H:%M %d/%m/%Y", tm);
  char hist[120];
  sprintf(hist, "[OUT] %s ← %s  (%s)", plate.c_str(), slotID, ts);
  addHistory(hist);
  saveHistory(hist);
  saveData();

  return "{\"ok\":true,\"msg\":\"Lấy xe thành công!\",\"slot\":\"" + std::string(slotID) + "\"}";
}

std::string handleReset()
{
  initLot();
  lot.carCount = 0;
  lot.historyCount = 0;
  saveData();

  const char *msg = "=== RESET BÃI XE ===";
  std::ofstream f("history.txt", std::ios::trunc);
  if (f)
    f << msg << "\n";
  addHistory(msg);

  return "{\"ok\":true,\"msg\":\"Đã reset bãi xe!\"}";
}
std::string handleScan(const std::string &body)
{
  // Nhận biển số từ Python ANPR script
  std::string plate = getParam(body, "plate");
  if (plate.empty())
    return "{\"ok\":false,\"msg\":\"ANPR: Không đọc được biển số\"}";

  // Gọi thẳng handlePark với plate đã nhận diện
  // Tái sử dụng body format giống /api/park
  return handlePark("plate=" + plate);
}

// Ghi file cờ hiệu để Python (anpr.py) biết là cần chụp & quét NGAY,
// thay vì tự động quét liên tục mỗi khi camera ghi đè current_frame.png.
std::string handleScanTrigger()
{
  if (!g_cameraOK)
    return "{\"ok\":false,\"msg\":\"Camera chưa kết nối, không thể chụp!\"}";

  std::ofstream f(SCAN_TRIGGER_FILE);
  if (!f)
    return "{\"ok\":false,\"msg\":\"Không ghi được file trigger!\"}";
  f << time(0); // nội dung không quan trọng, chỉ cần mtime file đổi
  f.close();

  return "{\"ok\":true,\"msg\":\"Đã gửi yêu cầu chụp & quét biển số!\"}";
}

// ============================================================
//  CAMERA FRAME SERVING  (đọc current_frame.png ra binary cho UI)
// ============================================================

bool readFileBinary(const std::string &path, std::string &out)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    return false;
  std::streamsize size = f.tellg();
  if (size <= 0)
    return false;
  f.seekg(0, std::ios::beg);
  out.resize((size_t)size);
  return (bool)f.read(&out[0], size);
}

// ============================================================
//  HTML PAGE
// ============================================================

const char *HTML = R"RAW(<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Parking AI Control</title>
<style>
  :root{
    --bg:#060916;
    --bg2:#0b1023;
    --card:rgba(18,25,48,.82);
    --card2:rgba(10,16,34,.92);
    --line:rgba(148,163,184,.16);
    --text:#e5edf8;
    --muted:#8ea0bd;
    --cyan:#22d3ee;
    --blue:#4f8cff;
    --green:#3cff9a;
    --red:#ff5470;
    --amber:#ffd166;
    --violet:#a78bfa;
    --shadow:0 22px 70px rgba(0,0,0,.45);
    --radius:22px;
  }

  *{box-sizing:border-box;margin:0;padding:0}
  html,body{min-height:100%}
  body{
    background:
      radial-gradient(circle at 10% 0%, rgba(34,211,238,.18), transparent 36%),
      radial-gradient(circle at 85% 12%, rgba(167,139,250,.16), transparent 36%),
      linear-gradient(135deg,var(--bg),var(--bg2));
    color:var(--text);
    font-family:Inter,Segoe UI,Arial,sans-serif;
    overflow-x:hidden;
  }

  body:before{
    content:"";
    position:fixed;inset:0;
    background-image:
      linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
      linear-gradient(90deg,rgba(255,255,255,.035) 1px, transparent 1px);
    background-size:48px 48px;
    mask-image:linear-gradient(to bottom,rgba(0,0,0,.7),transparent 85%);
    pointer-events:none;
  }

  .shell{position:relative;z-index:1;min-height:100vh;padding:22px}
  .topbar{
    height:72px;border:1px solid var(--line);background:rgba(10,16,34,.72);
    border-radius:var(--radius);box-shadow:var(--shadow);backdrop-filter:blur(18px);
    display:flex;align-items:center;justify-content:space-between;padding:0 22px;margin-bottom:18px;
  }
  .brand{display:flex;align-items:center;gap:14px}
  .brand-mark{
    width:46px;height:46px;border-radius:16px;display:grid;place-items:center;
    background:linear-gradient(135deg,var(--cyan),var(--blue));
    box-shadow:0 0 34px rgba(34,211,238,.32);
    color:#03111b;font-size:24px;font-weight:900;
  }
  .brand h1{font-size:20px;letter-spacing:.2px}
  .brand p{font-size:12px;color:var(--muted);margin-top:3px}
  .top-actions{display:flex;align-items:center;gap:12px}
  .pill{
    display:inline-flex;align-items:center;gap:8px;padding:9px 12px;border-radius:999px;
    border:1px solid var(--line);background:rgba(255,255,255,.045);color:var(--muted);
    font-size:12px;font-weight:700;
  }
  .dot{width:8px;height:8px;border-radius:50%;background:var(--red);box-shadow:0 0 18px currentColor}
  .pill.live{color:var(--green)}
  .pill.live .dot{background:var(--green);animation:pulse 1.15s infinite}
  @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(.72)}}

  .grid{display:grid;grid-template-columns:330px 1fr;gap:18px}
  .panel{
    border:1px solid var(--line);background:var(--card);border-radius:var(--radius);
    box-shadow:var(--shadow);backdrop-filter:blur(18px);overflow:hidden;
  }
  .side{padding:18px;display:flex;flex-direction:column;gap:16px;min-height:calc(100vh - 112px)}
  .label{
    color:var(--muted);font-size:11px;text-transform:uppercase;font-weight:900;
    letter-spacing:.16em;margin-bottom:10px;
  }

  .metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .metric{
    min-height:102px;border:1px solid var(--line);border-radius:18px;padding:14px;
    background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.025));
    position:relative;overflow:hidden;
  }
  .metric:after{
    content:"";position:absolute;right:-28px;top:-28px;width:80px;height:80px;border-radius:50%;
    background:rgba(34,211,238,.10);
  }
  .metric .num{font-size:34px;font-weight:900;line-height:1;font-variant-numeric:tabular-nums}
  .metric .txt{font-size:12px;color:var(--muted);margin-top:8px}
  .metric.free .num{color:var(--green)}
  .metric.busy .num{color:var(--red)}
  .metric.rate .num{color:var(--amber)}
  .metric.total .num{color:var(--cyan)}

  .tabs{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
  .tab{
    border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--muted);
    border-radius:14px;padding:10px;font-weight:900;cursor:pointer;transition:.18s;
  }
  .tab.active{background:linear-gradient(135deg,var(--cyan),var(--blue));color:#04111e;border-color:transparent}
  .input{
    width:100%;height:48px;border-radius:15px;border:1px solid var(--line);
    background:rgba(3,7,18,.72);color:var(--text);outline:none;padding:0 14px;
    font-weight:900;letter-spacing:.12em;text-transform:uppercase;font-size:15px;
  }
  .input:focus{border-color:rgba(34,211,238,.7);box-shadow:0 0 0 4px rgba(34,211,238,.08)}
  .btn{
    width:100%;height:48px;border:0;border-radius:15px;margin-top:10px;
    font-weight:950;cursor:pointer;transition:.18s;letter-spacing:.02em;
  }
  .btn:hover{transform:translateY(-1px);filter:brightness(1.06)}
  .btn:disabled{opacity:.55;cursor:not-allowed;transform:none}
  .btn-park{background:linear-gradient(135deg,var(--green),#10b981);color:#02130a}
  .btn-leave{background:linear-gradient(135deg,var(--red),#ef4444);color:white}
  .btn-scan{background:linear-gradient(135deg,#ff7a18,#ff3d81);color:white;height:54px}
  .btn-reset{background:transparent;color:var(--red);border:1px solid rgba(255,84,112,.42)}

  .slot-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
  .slot-chip{
    border:1px solid var(--line);border-radius:12px;padding:9px 0;text-align:center;
    font-size:12px;font-weight:950;font-variant-numeric:tabular-nums;background:rgba(255,255,255,.035)
  }
  .slot-chip.free{color:var(--green);border-color:rgba(60,255,154,.26)}
  .slot-chip.busy{color:var(--red);border-color:rgba(255,84,112,.30);background:rgba(255,84,112,.075)}

  main{display:flex;flex-direction:column;gap:18px}
  .hero{display:grid;grid-template-columns:1.08fr .92fr;gap:18px}
  .card-head{
    height:58px;display:flex;align-items:center;justify-content:space-between;
    padding:0 18px;border-bottom:1px solid var(--line)
  }
  .card-title{font-size:14px;font-weight:950;letter-spacing:.04em}
  .card-sub{font-size:12px;color:var(--muted)}
  .card-body{padding:18px}

  .camera-box{
    aspect-ratio:16/9;border-radius:18px;border:1px solid rgba(255,255,255,.09);
    background:
      radial-gradient(circle at center,rgba(34,211,238,.09),transparent 50%),
      #01040c;
    display:grid;place-items:center;overflow:hidden;position:relative;
  }
  .camera-box:before{
    content:"";position:absolute;inset:0;
    background:linear-gradient(transparent 48%,rgba(34,211,238,.10) 50%,transparent 52%);
    background-size:100% 46px;opacity:.25;pointer-events:none;
  }
  .camera-box img{width:100%;height:100%;object-fit:contain;display:none;position:relative;z-index:1}
  .cam-empty{text-align:center;color:var(--muted);font-weight:800}
  .cam-empty strong{display:block;color:var(--text);font-size:18px;margin-bottom:8px}
  .scan-row{display:grid;grid-template-columns:1fr;gap:10px;margin-top:14px}
  .scan-status{
    min-height:22px;color:var(--muted);font-size:12px;text-align:center;font-weight:700;
  }

  .map-wrap{display:grid;place-items:center}
  #map-canvas{width:100%;max-width:520px;height:auto}
  .path-info{
    display:none;margin-top:12px;border-radius:16px;border:1px solid rgba(34,211,238,.24);
    background:rgba(34,211,238,.07);padding:12px;color:var(--cyan);
    font-size:12px;line-height:1.45;font-weight:700;
  }
  .path-info.visible{display:block}

  .bottom{display:grid;grid-template-columns:1.2fr .8fr;gap:18px}
  .table-wrap{overflow:auto;max-height:330px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{
    position:sticky;top:0;background:rgba(10,16,34,.96);z-index:1;
    text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.12em;
    padding:12px;border-bottom:1px solid var(--line)
  }
  td{padding:13px 12px;border-bottom:1px solid rgba(148,163,184,.10)}
  tr:hover td{background:rgba(255,255,255,.035)}
  .plate-badge{
    display:inline-flex;padding:6px 10px;border-radius:10px;background:rgba(255,255,255,.06);
    border:1px solid var(--line);font-weight:950;letter-spacing:.08em
  }
  .slot-tag{font-weight:950;color:var(--cyan)}
  .time-tag{color:var(--muted);font-size:12px}
  .empty{
    color:var(--muted);text-align:center;padding:34px 12px;font-size:13px;font-weight:800;
  }

  .history{display:flex;flex-direction:column;gap:8px;max-height:330px;overflow:auto}
  .hist{
    border:1px solid var(--line);border-radius:14px;padding:11px 12px;
    background:rgba(255,255,255,.035);font-size:12px;font-weight:800;line-height:1.4;
  }
  .hist.in{color:var(--green);border-color:rgba(60,255,154,.20)}
  .hist.out{color:var(--red);border-color:rgba(255,84,112,.22)}
  .toast{
    position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(90px);
    z-index:50;background:rgba(10,16,34,.92);border:1px solid var(--line);border-radius:16px;
    box-shadow:var(--shadow);padding:13px 18px;font-weight:900;color:var(--text);
    transition:.28s;backdrop-filter:blur(14px);max-width:calc(100vw - 30px)
  }
  .toast.show{transform:translateX(-50%) translateY(0)}
  .toast.ok{border-color:rgba(60,255,154,.45);color:var(--green)}
  .toast.err{border-color:rgba(255,84,112,.45);color:var(--red)}

  @media(max-width:1050px){
    .grid{grid-template-columns:1fr}
    .side{min-height:auto}
    .hero,.bottom{grid-template-columns:1fr}
  }
</style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">P</div>
      <div>
        <h1>Smart Parking AI Control</h1>
        <p>YOLO ANPR · A* Pathfinding · Local C++ Web Server</p>
      </div>
    </div>
    <div class="top-actions">
      <span class="pill" id="cam-pill"><span class="dot"></span><span id="cam-text">Camera Offline</span></span>
      <span class="pill" id="clock">--:--:--</span>
    </div>
  </div>

  <div class="grid">
    <aside class="panel side">
      <section>
        <div class="label">Tổng quan bãi xe</div>
        <div class="metric-grid">
          <div class="metric total"><div class="num" id="s-total">—</div><div class="txt">Tổng slot</div></div>
          <div class="metric free"><div class="num" id="s-free">—</div><div class="txt">Còn trống</div></div>
          <div class="metric busy"><div class="num" id="s-busy">—</div><div class="txt">Đang đỗ</div></div>
          <div class="metric rate"><div class="num" id="s-rate">—</div><div class="txt">Tỉ lệ lấp đầy</div></div>
        </div>
      </section>

      <section class="panel" style="padding:14px;background:rgba(255,255,255,.025);box-shadow:none">
        <div class="label">Điều khiển thủ công</div>
        <div class="tabs">
          <button class="tab active" id="tab-park" onclick="switchTab('park')">Gửi xe</button>
          <button class="tab" id="tab-leave" onclick="switchTab('leave')">Lấy xe</button>
        </div>
        <input id="plate-in" class="input" placeholder="51A12345" maxlength="14" oninput="this.value=this.value.toUpperCase()">
        <button id="action-btn" class="btn btn-park" onclick="doAction()">Đưa xe vào bãi</button>
      </section>

      <section>
        <div class="label">Slot</div>
        <div class="slot-grid" id="slot-grid"></div>
      </section>

      <section style="margin-top:auto">
        <div class="label">Hệ thống</div>
        <button class="btn btn-reset" onclick="doReset()">Reset dữ liệu demo</button>
      </section>
    </aside>

    <main>
      <section class="hero">
        <div class="panel">
          <div class="card-head">
            <div>
              <div class="card-title">Camera trực tiếp</div>
              <div class="card-sub">C++ ghi frame → Python YOLO/OCR đọc khi bấm quét</div>
            </div>
            <span class="pill" id="scan-mode">Manual Scan</span>
          </div>
          <div class="card-body">
            <div class="camera-box">
              <img id="cam-feed" alt="camera">
              <div id="cam-placeholder" class="cam-empty"><strong>Đang chờ camera kết nối</strong>Kiểm tra server, webcam hoặc quyền camera</div>
            </div>
            <div class="scan-row">
              <button class="btn btn-scan" id="scan-btn" onclick="doScanTrigger()">Chụp & quét biển số bằng AI</button>
              <div class="scan-status" id="scan-status">Sẵn sàng nhận tín hiệu quét.</div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="card-head">
            <div>
              <div class="card-title">Sơ đồ bãi xe + A*</div>
              <div class="card-sub">Tự chọn slot gần nhất và vẽ đường đi</div>
            </div>
            <span class="pill">5 × 8 Grid</span>
          </div>
          <div class="card-body">
            <div class="map-wrap"><canvas id="map-canvas" width="520" height="340"></canvas></div>
            <div class="path-info" id="path-info"></div>
          </div>
        </div>
      </section>

      <section class="bottom">
        <div class="panel">
          <div class="card-head">
            <div>
              <div class="card-title">Xe đang trong bãi</div>
              <div class="card-sub" id="car-count">0 xe</div>
            </div>
            <span class="pill">Realtime</span>
          </div>
          <div class="table-wrap">
            <table>
              <thead><tr><th>#</th><th>Biển số</th><th>Slot</th><th>Giờ vào</th></tr></thead>
              <tbody id="cars-tbody"></tbody>
            </table>
          </div>
        </div>

        <div class="panel">
          <div class="card-head">
            <div>
              <div class="card-title">Lịch sử giao dịch</div>
              <div class="card-sub">20 giao dịch gần nhất</div>
            </div>
          </div>
          <div class="card-body"><div class="history" id="history-list"></div></div>
        </div>
      </section>
    </main>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const MAP = [[0,0,0,0,2,0,2,0],[0,1,1,0,2,0,2,0],[2,0,0,0,2,0,2,0],[2,1,1,0,2,0,2,0],[2,2,2,2,2,2,2,2]];
let state = {tab:'park', path:[], target:null, lastCars:0};

function clock(){
  document.getElementById('clock').textContent = new Date().toLocaleTimeString('vi-VN',{hour12:false});
}
setInterval(clock,1000); clock();

function toast(msg, ok=true){
  const t=document.getElementById('toast');
  t.textContent=msg; t.className='toast show '+(ok?'ok':'err');
  setTimeout(()=>t.className='toast',2800);
}

function switchTab(tab){
  state.tab=tab;
  document.getElementById('tab-park').className='tab '+(tab==='park'?'active':'');
  document.getElementById('tab-leave').className='tab '+(tab==='leave'?'active':'');
  const btn=document.getElementById('action-btn');
  btn.className='btn '+(tab==='park'?'btn-park':'btn-leave');
  btn.textContent=tab==='park'?'Đưa xe vào bãi':'Cho xe rời bãi';
}

async function getStatus(){
  const r=await fetch('/api/status');
  return await r.json();
}

async function refresh(){
  try{
    const d=await getStatus();
    renderStatus(d);
  }catch(e){}
}

function renderStatus(d){
  document.getElementById('s-total').textContent=d.total;
  document.getElementById('s-free').textContent=d.free;
  document.getElementById('s-busy').textContent=d.occupied;
  document.getElementById('s-rate').textContent=(d.total?Math.round(d.occupied/d.total*100):0)+'%';

  const pill=document.getElementById('cam-pill');
  const txt=document.getElementById('cam-text');
  if(d.camera){ pill.className='pill live'; txt.textContent='Camera Live'; }
  else { pill.className='pill'; txt.textContent='Camera Offline'; }

  const sg=document.getElementById('slot-grid');
  sg.innerHTML='';
  d.slots.forEach(s=>{
    const el=document.createElement('div');
    el.className='slot-chip '+(s.occupied?'busy':'free');
    el.textContent=s.id;
    sg.appendChild(el);
  });

  const tb=document.getElementById('cars-tbody');
  document.getElementById('car-count').textContent=d.cars.length+' xe';
  if(!d.cars.length){
    tb.innerHTML='<tr><td colspan="4" class="empty">Không có xe nào trong bãi</td></tr>';
  }else{
    tb.innerHTML=d.cars.map((c,i)=>`
      <tr>
        <td>${i+1}</td>
        <td><span class="plate-badge">${escapeHtml(c.plate)}</span></td>
        <td><span class="slot-tag">${escapeHtml(c.slot)}</span></td>
        <td><span class="time-tag">${escapeHtml(c.checkin)}</span></td>
      </tr>`).join('');
  }

  const h=document.getElementById('history-list');
  if(!d.history.length) h.innerHTML='<div class="empty">Chưa có giao dịch</div>';
  else h.innerHTML=d.history.map(x=>{
    const isIn=x.startsWith('[IN');
    return `<div class="hist ${isIn?'in':'out'}">${isIn?'↓':'↑'} ${escapeHtml(x)}</div>`;
  }).join('');

  drawMap(d.slots,state.path,state.target);
  state.lastCars=d.cars.length;
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));
}

function drawMap(slots,path,target){
  const c=document.getElementById('map-canvas'), ctx=c.getContext('2d');
  const rows=5, cols=8, cw=c.width/cols, ch=c.height/rows;
  ctx.clearRect(0,0,c.width,c.height);
  const slotMap={}; slots.forEach(s=>slotMap[s.x+','+s.y]=s);
  const pathSet=new Set(path.map(p=>p[0]+','+p[1]));

  for(let r=0;r<rows;r++){
    for(let col=0;col<cols;col++){
      const x=col*cw,y=r*ch,key=r+','+col,v=MAP[r][col];
      const grad=ctx.createLinearGradient(x,y,x+cw,y+ch);
      if(v===1){ grad.addColorStop(0,'#201525'); grad.addColorStop(1,'#0f0a18'); }
      else if(v===2 && slotMap[key]?.occupied){ grad.addColorStop(0,'#401020'); grad.addColorStop(1,'#1b0710'); }
      else if(v===2){ grad.addColorStop(0,'#0b3b28'); grad.addColorStop(1,'#071d17'); }
      else { grad.addColorStop(0,'#10203b'); grad.addColorStop(1,'#071022'); }
      ctx.fillStyle=grad; roundRect(ctx,x+5,y+5,cw-10,ch-10,14,true,false);
      ctx.strokeStyle='rgba(148,163,184,.17)'; ctx.lineWidth=1; roundRect(ctx,x+5,y+5,cw-10,ch-10,14,false,true);

      if(pathSet.has(key)&&v!==1){
        ctx.fillStyle='rgba(34,211,238,.18)'; roundRect(ctx,x+5,y+5,cw-10,ch-10,14,true,false);
        ctx.fillStyle='#22d3ee'; ctx.beginPath(); ctx.arc(x+cw/2,y+ch/2,4,0,Math.PI*2); ctx.fill();
      }
      if(v===2 && slotMap[key]){
        const s=slotMap[key];
        const hit=target && target[0]===r && target[1]===col;
        ctx.font='900 15px Segoe UI'; ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillStyle=hit?'#3cff9a':(s.occupied?'#ff5470':'#22d3ee');
        ctx.fillText(s.id,x+cw/2,y+ch/2-2);
        if(s.occupied){ctx.font='17px Segoe UI Emoji';ctx.fillText('🚗',x+cw/2,y+ch/2+20);}
      }
      if(r===0 && col===0){ctx.font='25px Segoe UI Emoji';ctx.fillText('🚦',x+cw/2,y+ch/2);}
    }
  }
  if(path.length>1){
    ctx.beginPath();
    ctx.moveTo(path[0][1]*cw+cw/2,path[0][0]*ch+ch/2);
    path.slice(1).forEach(p=>ctx.lineTo(p[1]*cw+cw/2,p[0]*ch+ch/2));
    ctx.strokeStyle='#22d3ee'; ctx.lineWidth=4; ctx.setLineDash([10,7]); ctx.stroke(); ctx.setLineDash([]);
  }
}

function roundRect(ctx,x,y,w,h,r,fill,stroke){
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
  if(fill)ctx.fill(); if(stroke)ctx.stroke();
}

async function doAction(){
  const plate=document.getElementById('plate-in').value.trim();
  if(!plate) return toast('Vui lòng nhập biển số xe',false);
  const btn=document.getElementById('action-btn'); btn.disabled=true; btn.textContent='Đang xử lý...';
  try{
    const url=state.tab==='park'?'/api/park':'/api/leave';
    const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'plate='+encodeURIComponent(plate)});
    const d=await r.json();
    toast(d.msg,d.ok);
    if(d.ok && state.tab==='park' && d.path){
      state.path=d.path; state.target=[d.x,d.y];
      const pi=document.getElementById('path-info');
      pi.className='path-info visible';
      pi.textContent=`A* chọn ${d.slot}: ${d.steps} bước · ${d.path.map(p=>'('+p[0]+','+p[1]+')').join(' → ')}`;
    }
    if(d.ok && state.tab==='leave'){state.path=[];state.target=null;document.getElementById('path-info').className='path-info';}
    if(d.ok) document.getElementById('plate-in').value='';
    await refresh();
  }finally{btn.disabled=false;switchTab(state.tab);}
}

async function doReset(){
  if(!confirm('Reset toàn bộ dữ liệu demo?')) return;
  const d=await (await fetch('/api/reset',{method:'POST'})).json();
  state.path=[];state.target=null;document.getElementById('path-info').className='path-info';
  toast(d.msg,d.ok); refresh();
}

async function doScanTrigger(){
  const btn=document.getElementById('scan-btn'), st=document.getElementById('scan-status');
  btn.disabled=true; btn.textContent='Đang gửi tín hiệu...'; st.textContent='Đang yêu cầu C++ chụp frame hiện tại...';
  try{
    const before=(await getStatus()).cars.length;
    const d=await (await fetch('/api/scan-trigger',{method:'POST'})).json();
    if(!d.ok){st.textContent=d.msg;toast(d.msg,false);return;}
    st.textContent='Đã gửi trigger. Đang chờ Python YOLO + OCR trả kết quả...';
    let tries=0;
    const timer=setInterval(async()=>{
      tries++;
      const s=await getStatus();
      renderStatus(s);
      if(s.cars.length>before){
        clearInterval(timer);
        st.textContent='Đã nhận diện và thêm xe vào bãi.';
        toast('Quét biển số thành công',true);
      }else if(tries>=18){
        clearInterval(timer);
        st.textContent='Chưa nhận được kết quả. Kiểm tra anpr.py có đang chạy không.';
      }
    },1000);
  }catch(e){st.textContent='Lỗi kết nối server';toast('Lỗi kết nối server',false);}
  finally{btn.disabled=false;btn.textContent='Chụp & quét biển số bằng AI';}
}

function updateFrame(){
  const img=document.getElementById('cam-feed'), ph=document.getElementById('cam-placeholder');
  const probe=new Image();
  probe.onload=()=>{img.src=probe.src;img.style.display='block';ph.style.display='none';};
  probe.onerror=()=>{img.style.display='none';ph.style.display='block';};
  probe.src='/api/frame?t='+Date.now();
}

document.getElementById('plate-in').addEventListener('keydown',e=>{if(e.key==='Enter')doAction();});
refresh(); updateFrame();
setInterval(refresh,2500);
setInterval(updateFrame,650);
</script>
</body>
</html>
)RAW";

// ============================================================
//  HTTP SERVER
// ============================================================

std::string httpResponse(int code, const std::string &ct, const std::string &body)
{
  std::string status = (code == 200) ? "200 OK" : (code == 404) ? "404 Not Found"
                                                                : "400 Bad Request";
  std::ostringstream r;
  r << "HTTP/1.1 " << status << "\r\n"
    << "Content-Type: " << ct << "\r\n"
    << "Content-Length: " << body.size() << "\r\n"
    << "Access-Control-Allow-Origin: *\r\n"
    << "Connection: close\r\n\r\n"
    << body;
  return r.str();
}

void handleClient(int sock)
{
  char buf[8192] = {};
  recv(sock, buf, sizeof(buf) - 1, 0);

  std::string req(buf);
  // parse method + path
  std::string method, path, bodyStr;
  {
    std::istringstream ss(req);
    std::string version;
    ss >> method >> path >> version;
  }

  // find body (after \r\n\r\n)
  size_t bp = req.find("\r\n\r\n");
  if (bp != std::string::npos)
    bodyStr = req.substr(bp + 4);

  std::string resp;
  if (path == "/" || path == "/index.html")
  {
    resp = httpResponse(200, "text/html; charset=utf-8", std::string(HTML));
  }
  else if (path.rfind("/api/frame", 0) == 0 && method == "GET")
  {
    std::string jpg;
    if (readFileBinary(CURRENT_FRAME_FILE, jpg))
    {
      std::ostringstream r;
      r << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: image/bmp\r\n"
        << "Content-Length: " << jpg.size() << "\r\n"
        << "Cache-Control: no-store, no-cache, must-revalidate\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Connection: close\r\n\r\n";
      resp = r.str();
      resp.append(jpg); // append binary an toàn (giữ nguyên byte null nếu có)
    }
    else
    {
      resp = httpResponse(404, "text/plain", "No frame");
    }
  }
  else if (path == "/api/status" && method == "GET")
  {
    resp = httpResponse(200, "application/json", lotStatusJSON());
  }
  else if (path == "/api/park" && method == "POST")
  {
    resp = httpResponse(200, "application/json", handlePark(bodyStr));
  }
  else if (path == "/api/leave" && method == "POST")
  {
    resp = httpResponse(200, "application/json", handleLeave(bodyStr));
  }
  else if (path == "/api/reset" && method == "POST")
  {
    resp = httpResponse(200, "application/json", handleReset());
  }
  else if (path == "/api/scan" && method == "POST")
  {
    resp = httpResponse(200, "application/json", handleScan(bodyStr));
  }
  else if (path == "/api/scan-trigger" && method == "POST")
  {
    resp = httpResponse(200, "application/json", handleScanTrigger());
  }
  else
  {
    resp = httpResponse(404, "text/plain", "Not found");
  }

  send(sock, resp.c_str(), resp.size(), 0);
  CLOSE(sock);
}

int main()
{
#ifdef _WIN32
  WSADATA wd;
  WSAStartup(MAKEWORD(2, 2), &wd);
#endif

  initLot();
  loadData();
  loadHistory();

  int server = socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0)
  {
    std::cerr << "Cannot create socket\n";
    return 1;
  }

  int opt = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(8080);

  if (bind(server, (sockaddr *)&addr, sizeof(addr)) < 0)
  {
    std::cerr << "Bind failed (port 8080 busy?)\n";
    return 1;
  }
  listen(server, 16);
  std::cout << "Starting server..." << std::endl;

  // Khởi chạy luồng chụp ảnh camera song song với HTTP server.
  // Luồng này ghi liên tục current_frame.png để anpr.py (Python) đọc và xử lý AI.
  std::thread camThread(cameraCaptureLoop);
  camThread.detach();

  system("start http://localhost:8080");

  while (true)
  {
    sockaddr_in cli{};
    socklen_t clen = sizeof(cli);
    int client = accept(server, (sockaddr *)&cli, &clen);
    if (client < 0)
      continue;
    handleClient(client);
  }

  g_cameraRunning = false;
  CLOSE(server);
  return 0;
}