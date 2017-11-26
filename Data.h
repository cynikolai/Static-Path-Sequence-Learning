#include <vector>
#include <string>
#include <fstream>

struct BBData {
  std::vector<std::string> opcodes;
};

struct PathData {
  double frequency;
  std::vector<BBData> bbdata;
};

std::ofstream& operator<<(std::ofstream& os, const BBData &bb);

std::ofstream& operator<<(std::ofstream &os, const PathData& pd);
