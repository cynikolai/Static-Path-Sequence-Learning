#define DEBUG_TYPE "pathminer"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/PathProfileInfo.h"
#include "llvm/Analysis/Passes.h"
#include "Data.h"
#include <iostream>
#include <fstream>

using namespace llvm;

namespace {
  struct PathMiner : public FunctionPass {
    static char ID;
    std::ofstream fout;
    PathMiner() : FunctionPass(ID) {
      fout.open("output.xml");
    }
    ~PathMiner() {
      fout.close();
    }

    bool runOnFunction(Function &F) {

      ModulePass* mp = createPathProfileLoaderPass();
      mp->runOnModule(*F.getParent());

      AnalysisID id = &PathProfileInfo::ID;
      PathProfileInfo* ppi = (PathProfileInfo*) mp->getAdjustedAnalysisPointer(id);

      ppi->setCurrentFunction(&F);
      processPaths(ppi, fout);

      //      fout.close();

      return true;
    }

    void processPaths(PathProfileInfo* ppi, std::ofstream &os) {
      if(ppi->pathsRun() == 0) {
	return;
      }
      for (ProfilePathIterator it = ppi->pathBegin(); it != ppi->pathEnd(); it++) {
	os << getPathData(it->second);
      }
    }

    PathData getPathData(ProfilePath* pp) {
      PathData pathData;

      pathData.frequency = pp->getFrequency() / 100.0;

      ProfilePathBlockVector* ppbv = pp->getPathBlocks();
      for(int i = 0; i < ppbv->size(); i++) {
	BBData bbdata = getBBData((*ppbv)[i]);
	pathData.bbdata.push_back(bbdata);
      }

      return pathData;
    }

    BBData getBBData(BasicBlock* bb) {
      BBData bbdata;
      if(bb->size() > 0) {
	for(BasicBlock::iterator it = bb->begin(); it != bb->end(); it++) {
	  bbdata.opcodes.push_back(std::string(it->getOpcodeName()));
	}
      }
      return bbdata;
    }
  };
}

char PathMiner::ID = 0;
static RegisterPass<PathMiner> X("pathminer", "Path Miner Pass");
