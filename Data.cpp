#include <vector>
#include <string>
#include <fstream>
#include "Data.h"

std::ofstream& operator<<(std::ofstream& os, const BBData &bb) {
  os << "\t<basic_block>" << std::endl;
  for(int i = 0; i < bb.opcodes.size(); i++) {
    os << "\t\t<opcode>" + bb.opcodes[i] + "</opcode>" << std::endl;
  }
  os << "\t</basic_block>" << std::endl;
  return os;
}

std::ofstream& operator<<(std::ofstream &os, const PathData& pd) {
  os << "<path>" << std::endl;
  for(int i = 0; i < pd.bbdata.size(); i++) {
    os << pd.bbdata[i];
  }
  os << "\t<frequency>" << std::endl;
  os << "\t\t" << pd.frequency << std::endl;
  os << "\t</frequency>" << std::endl;
  os << "</path>" << std::endl;
}
