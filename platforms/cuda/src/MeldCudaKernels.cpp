/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/


#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "MeldCudaKernels.h"
#include "CudaMeldKernelSources.h"
#include "openmm/internal/ContextImpl.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include <sys/time.h>

#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<errorMessage<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

CudaCalcMeldForceKernel::CudaCalcMeldForceKernel(std::string name, const Platform& platform, CudaContext& cu,
                                                 const System& system) :
    CalcMeldForceKernel(name, platform), cu(cu), system(system)
{

    if (cu.getUseDoublePrecision()) {
        cout << "***\n";
        cout << "*** MeldForce does not support double precision.\n";
        cout << "***" << endl;
        throw OpenMMException("MeldForce does not support double precision");
    }

    numDistRestraints = 0;
    numHyperbolicDistRestraints = 0;
    numTorsionRestraints = 0;
    numDistProfileRestraints = 0;
    numRestraints = 0;
    numGroups = 0;
    numCollections = 0;
    ecoCutoff = 0;
    numResidues = 0;
    largestGroup = 0;
    largestCollection = 0;
    groupsPerBlock = -1;

    distanceRestRParams = NULL;
    distanceRestKParams = NULL;
    distanceRestDoingEco = NULL;
    distanceRestEcoFactors = NULL;
    distanceRestEcoConstants = NULL;
    distanceRestEcoLinears = NULL;
    distanceRestEcoValues = NULL;
    //distanceRestCOValues = NULL;
    distanceRestContacts = NULL;
    distanceRestEdgeCounts = NULL;
    alphaCarbons = NULL;
    dijkstra_unexplored = NULL;
    dijkstra_unexplored_old = NULL;
    dijkstra_frontier = NULL;
    dijkstra_frontier_old = NULL;
    dijkstra_distance = NULL;
    dijkstra_n_explored = NULL;
    dijkstra_total = NULL;
    //alphaCarbonPosq = NULL;
    distanceRestAtomIndices = NULL;
    distanceRestResidueIndices = NULL;
    distanceRestGlobalIndices = NULL;
    distanceRestForces = NULL;
    hyperbolicDistanceRestRParams = NULL;
    hyperbolicDistanceRestParams = NULL;
    hyperbolicDistanceRestAtomIndices = NULL;
    hyperbolicDistanceRestGlobalIndices = NULL;
    hyperbolicDistanceRestForces = NULL;
    torsionRestParams = NULL;
    torsionRestAtomIndices = NULL;
    torsionRestGlobalIndices = NULL;
    torsionRestForces = NULL;
    distProfileRestAtomIndices = NULL;
    distProfileRestDistRanges = NULL;
    distProfileRestNumBins = NULL;
    distProfileRestParamBounds = NULL;
    distProfileRestParams = NULL;
    distProfileRestScaleFactor = NULL;
    distProfileRestGlobalIndices = NULL;
    distProfileRestForces = NULL;
    torsProfileRestAtomIndices0 = NULL;
    torsProfileRestAtomIndices1 = NULL;
    torsProfileRestNumBins = NULL;
    torsProfileRestParamBounds = NULL;
    torsProfileRestParams0 = NULL;
    torsProfileRestParams1 = NULL;
    torsProfileRestParams2 = NULL;
    torsProfileRestParams3 = NULL;
    torsProfileRestScaleFactor = NULL;
    torsProfileRestGlobalIndices = NULL;
    torsProfileRestForces = NULL;
    restraintEnergies = NULL;
    //nonECOrestraintEnergies = NULL;
    restraintActive = NULL;
    groupRestraintIndices = NULL;
    groupRestraintIndicesTemp = NULL;
    groupEnergies = NULL;
    groupActive = NULL;
    groupBounds = NULL;
    groupNumActive = NULL;
    collectionGroupIndices = NULL;
    collectionBounds = NULL;
    collectionNumActive = NULL;
    collectionEnergies = NULL;
}

CudaCalcMeldForceKernel::~CudaCalcMeldForceKernel() {
    cu.setAsCurrent();
    delete distanceRestRParams;
    delete distanceRestKParams;
    delete distanceRestDoingEco;
    delete distanceRestEcoFactors;
    delete distanceRestEcoConstants;
    delete distanceRestEcoLinears;
    delete distanceRestEcoValues;
    //delete distanceRestCOValues;
    delete distanceRestContacts;
    delete distanceRestEdgeCounts;
    delete alphaCarbons;
    delete dijkstra_unexplored;
    delete dijkstra_unexplored_old;
    delete dijkstra_frontier;
    delete dijkstra_frontier_old;
    delete dijkstra_distance;
    delete dijkstra_n_explored;
    delete dijkstra_total;
    //delete alphaCarbonPosq;
    delete distanceRestAtomIndices;
    delete distanceRestResidueIndices;
    delete distanceRestGlobalIndices;
    delete distanceRestForces;
    delete hyperbolicDistanceRestRParams;
    delete hyperbolicDistanceRestParams;
    delete hyperbolicDistanceRestAtomIndices;
    delete hyperbolicDistanceRestGlobalIndices;
    delete hyperbolicDistanceRestForces;
    delete torsionRestParams;
    delete torsionRestAtomIndices;
    delete torsionRestGlobalIndices;
    delete torsionRestForces;
    delete distProfileRestAtomIndices;
    delete distProfileRestDistRanges;
    delete distProfileRestNumBins;
    delete distProfileRestParamBounds;
    delete distProfileRestParams;
    delete distProfileRestScaleFactor;
    delete distProfileRestGlobalIndices;
    delete distProfileRestForces;
    delete torsProfileRestAtomIndices0;
    delete torsProfileRestAtomIndices1;
    delete torsProfileRestNumBins;
    delete torsProfileRestParamBounds;
    delete torsProfileRestParams0;
    delete torsProfileRestParams1;
    delete torsProfileRestParams2;
    delete torsProfileRestParams3;
    delete torsProfileRestScaleFactor;
    delete torsProfileRestGlobalIndices;
    delete torsProfileRestForces;
    delete restraintEnergies;
    //delete nonECOrestraintEnergies;
    delete restraintActive;
    delete groupRestraintIndices;
    delete groupRestraintIndicesTemp;
    delete groupEnergies;
    delete groupActive;
    delete groupBounds;
    delete groupNumActive;
    delete collectionBounds;
    delete collectionNumActive;
    delete collectionEnergies;
}


void CudaCalcMeldForceKernel::allocateMemory(const MeldForce& force) {
    numDistRestraints = force.getNumDistRestraints();
    numHyperbolicDistRestraints = force.getNumHyperbolicDistRestraints();
    numTorsionRestraints = force.getNumTorsionRestraints();
    numDistProfileRestraints = force.getNumDistProfileRestraints();
    numDistProfileRestParams = force.getNumDistProfileRestParams();
    numTorsProfileRestraints = force.getNumTorsProfileRestraints();
    numTorsProfileRestParams = force.getNumTorsProfileRestParams();
    numRestraints = force.getNumTotalRestraints();
    numGroups = force.getNumGroups();
    numCollections = force.getNumCollections();
    ecoCutoff = force.getEcoCutoff();
    numResidues = force.getNumResidues();
    INF = 9999;
    timevar = 0;
    timecount = 0;

    // setup device memory
    if (numDistRestraints > 0) {
        distanceRestRParams        = CudaArray::create<float4> ( cu, numDistRestraints, "distanceRestRParams");
        distanceRestKParams        = CudaArray::create<float>  ( cu, numDistRestraints, "distanceRestKParams");
        distanceRestDoingEco       = CudaArray::create<int>    ( cu, numDistRestraints, "distanceRestDoingEco");
        distanceRestEcoFactors     = CudaArray::create<float>  ( cu, numDistRestraints, "distanceRestEcoFactors");
        distanceRestEcoConstants   = CudaArray::create<float>  ( cu, numDistRestraints, "distanceRestEcoConstants");
        distanceRestEcoLinears     = CudaArray::create<float>  ( cu, numDistRestraints, "distanceRestEcoLinears");
        distanceRestEcoValues      = CudaArray::create<float>  ( cu, numDistRestraints, "distanceRestEcoValues");
        //distanceRestCOValues       = CudaArray::create<float>  ( cu, numDistRestraints, "distanceRestCOValues");
	distanceRestContacts             = CudaArray::create<int>    ( cu, numResidues*numResidues, "distanceRestContacts");
	distanceRestEdgeCounts           = CudaArray::create<int>    ( cu, numResidues, "distanceRestEdgeCounts");
        distanceRestAtomIndices    = CudaArray::create<int2>   ( cu, numDistRestraints, "distanceRestAtomIndices");
  distanceRestResidueIndices       = CudaArray::create<int2>   ( cu, numDistRestraints, "distanceRestResidueIndices");
        distanceRestGlobalIndices  = CudaArray::create<int>    ( cu, numDistRestraints, "distanceRestGlobalIndices");
        distanceRestForces         = CudaArray::create<float3> ( cu, numDistRestraints, "distanceRestForces");
	alphaCarbons		                 = CudaArray::create<int>    ( cu, numResidues, "alphaCarbons");
	dijkstra_unexplored              = CudaArray::create<bool>   ( cu, numResidues, "dijkstra_unexplored");
  dijkstra_unexplored_old          = CudaArray::create<bool>   ( cu, numResidues, "dijkstra_unexplored_old");
  dijkstra_frontier                = CudaArray::create<bool>   ( cu, numResidues, "dijkstra_frontier");
  dijkstra_frontier_old            = CudaArray::create<bool>   ( cu, numResidues, "dijkstra_frontier_old");
  dijkstra_distance                = CudaArray::create<int>    ( cu, numResidues, "dijkstra_distance");
  dijkstra_n_explored              = CudaArray::create<int>    ( cu, numResidues, "dijkstra_n_explored");
  dijkstra_total                   = CudaArray::create<int>    ( cu, 1, "dijkstra_total");
  //alphaCarbonPosq                  = CudaArray::create<float>  ( cu, numResidues*3, "alphaCarbonPosq");
    }

    if (numHyperbolicDistRestraints > 0) {
        hyperbolicDistanceRestRParams        = CudaArray::create<float4> ( cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestRParams");
        hyperbolicDistanceRestParams         = CudaArray::create<float4> ( cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestParams");
        hyperbolicDistanceRestAtomIndices    = CudaArray::create<int2>   ( cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestAtomIndices");
        hyperbolicDistanceRestGlobalIndices  = CudaArray::create<int>    ( cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestGlobalIndices");
        hyperbolicDistanceRestForces         = CudaArray::create<float3> ( cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestForces");
    }

    if (numTorsionRestraints > 0) {
        torsionRestParams          = CudaArray::create<float3> (cu, numTorsionRestraints, "torsionRestParams");
        torsionRestAtomIndices     = CudaArray::create<int4>   (cu, numTorsionRestraints, "torsionRestAtomIndices");
        torsionRestGlobalIndices   = CudaArray::create<int>    (cu, numTorsionRestraints, "torsionRestGlobalIndices");
        torsionRestForces          = CudaArray::create<float3> (cu, numTorsionRestraints * 4, "torsionRestForces");
    }

    if (numDistProfileRestraints > 0) {
        distProfileRestAtomIndices = CudaArray::create<int2>   (cu, numDistProfileRestraints, "distProfileRestAtomIndices");
        distProfileRestDistRanges  = CudaArray::create<float2> (cu, numDistProfileRestraints, "distProfileRestDistRanges");
        distProfileRestNumBins     = CudaArray::create<int>    (cu, numDistProfileRestraints, "distProfileRestNumBins");
        distProfileRestParamBounds = CudaArray::create<int2>   (cu, numDistProfileRestraints, "distProfileRestParamBounds");
        distProfileRestParams      = CudaArray::create<float4> (cu, numDistProfileRestParams, "distProfileRestParams");
        distProfileRestScaleFactor = CudaArray::create<float>  (cu, numDistProfileRestraints, "distProfileRestScaleFactor");
        distProfileRestGlobalIndices=CudaArray::create<int>    (cu, numDistProfileRestraints, "distProfileRestGlobalIndices");
        distProfileRestForces      = CudaArray::create<float3> (cu, numDistProfileRestraints, "distProfileRestForces");
    }

    if (numTorsProfileRestraints > 0) {
        torsProfileRestAtomIndices0= CudaArray::create<int4>   (cu, numTorsProfileRestraints, "torsProfileRestAtomIndices0");
        torsProfileRestAtomIndices1= CudaArray::create<int4>   (cu, numTorsProfileRestraints, "torsProfileRestAtomIndices1");
        torsProfileRestNumBins     = CudaArray::create<int>    (cu, numTorsProfileRestraints, "torsProfileRestNumBins");
        torsProfileRestParamBounds = CudaArray::create<int2>   (cu, numTorsProfileRestraints, "torsProfileRestParamBounds");
        torsProfileRestParams0     = CudaArray::create<float4> (cu, numTorsProfileRestParams, "torsProfileRestParams0");
        torsProfileRestParams1     = CudaArray::create<float4> (cu, numTorsProfileRestParams, "torsProfileRestParams1");
        torsProfileRestParams2     = CudaArray::create<float4> (cu, numTorsProfileRestParams, "torsProfileRestParams2");
        torsProfileRestParams3     = CudaArray::create<float4> (cu, numTorsProfileRestParams, "torsProfileRestParams3");
        torsProfileRestScaleFactor = CudaArray::create<float>  (cu, numTorsProfileRestraints, "torsProfileRestScaleFactor");
        torsProfileRestGlobalIndices=CudaArray::create<int>    (cu, numTorsProfileRestraints, "torsProfileRestGlobalIndices");
        torsProfileRestForces      = CudaArray::create<float3> (cu, 8 * numTorsProfileRestraints, "torsProfileRestForces");
    }

    restraintEnergies         = CudaArray::create<float>  ( cu, numRestraints,     "restraintEnergies");
    //nonECOrestraintEnergies   = CudaArray::create<float>  ( cu, numRestraints,     "nonECOrestraintEnergies");
    restraintActive           = CudaArray::create<float>  ( cu, numRestraints,     "restraintActive");
    groupRestraintIndices     = CudaArray::create<int>    ( cu, numRestraints,     "groupRestraintIndices");
    groupRestraintIndicesTemp = CudaArray::create<int>    ( cu, numRestraints,     "groupRestraintIndicesTemp");
    groupEnergies             = CudaArray::create<float>  ( cu, numGroups,         "groupEnergies");
    groupActive               = CudaArray::create<float>  ( cu, numGroups,         "groupActive");
    groupBounds               = CudaArray::create<int2>   ( cu, numGroups,         "groupBounds");
    groupNumActive            = CudaArray::create<int>    ( cu, numGroups,         "groupNumActive");
    collectionGroupIndices    = CudaArray::create<int>    ( cu, numGroups,         "collectionGroupIndices");
    collectionBounds          = CudaArray::create<int2>   ( cu, numCollections,    "collectionBounds");
    collectionNumActive       = CudaArray::create<int>    ( cu, numCollections,    "collectionNumActive");
    collectionEnergies        = CudaArray::create<int>    ( cu, numCollections,    "collectionEnergies");

    // setup host memory
    h_distanceRestRParams                 = std::vector<float4> (numDistRestraints, make_float4( 0, 0, 0, 0));
    h_distanceRestKParams                 = std::vector<float>  (numDistRestraints, 0);
    h_distanceRestDoingEco                = std::vector<int>    (numDistRestraints, 0);
    h_distanceRestEcoFactors              = std::vector<float>  (numDistRestraints, 0);
    h_distanceRestEcoConstants            = std::vector<float>  (numDistRestraints, 0);
    h_distanceRestEcoLinears              = std::vector<float>  (numDistRestraints, 0);
    //h_distanceRestCOValues                = std::vector<float>  (numDistRestraints, 0);
    h_alphaCarbons                        = std::vector<int>    (numResidues, 0);
    h_distRestSorted                      = std::vector<int>    (numDistRestraints * 3, 0);
h_restraintEnergies                       = std::vector<float>    (numRestraints, 0);
h_restraintNonEcoEnergies                       = std::vector<float>    (numRestraints, 0);
h_distanceRestContacts                    = std::vector<int>    (numResidues*numResidues, 0);
h_distanceRestEdgeCounts                  = std::vector<int>    (numResidues, 0);
h_dijkstra_total                          = std::vector<int>    (1, 0);
h_dijkstra_distance                       = std::vector<int>    (numResidues, 0);
h_dijkstra_distance2                      = std::vector<int>    (numResidues, 0);
h_distanceRestEcoValues                   = std::vector<float>  (numDistRestraints, 0);
//h_alphaCarbonPosq                         = std::vector<float>  (numResidues*3, 0);
h_dijkstra_unexplored                     = std::vector<int>    (numResidues, 0);
h_dijkstra_unexplored_old                 = std::vector<int>    (numResidues, 0);
h_dijkstra_frontier                       = std::vector<int>    (numResidues, 0);
h_dijkstra_frontier_old                   = std::vector<int>    (numResidues, 0);
h_dijkstra_n_explored                     = std::vector<int>    (numResidues, 0);
h_dijkstra_n_explored_old                 = std::vector<int>    (numResidues, 0);
    h_distanceRestAtomIndices             = std::vector<int2>   (numDistRestraints, make_int2( -1, -1));
    h_distanceRestResidueIndices          = std::vector<int2>   (numDistRestraints, make_int2( -1, -1));
    h_distanceRestGlobalIndices           = std::vector<int>    (numDistRestraints, -1);
    h_hyperbolicDistanceRestRParams       = std::vector<float4> (numHyperbolicDistRestraints, make_float4( 0, 0, 0, 0));
    h_hyperbolicDistanceRestParams        = std::vector<float4> (numHyperbolicDistRestraints, make_float4( 0, 0, 0, 0));
    h_hyperbolicDistanceRestAtomIndices   = std::vector<int2>   (numHyperbolicDistRestraints, make_int2( -1, -1));
    h_hyperbolicDistanceRestGlobalIndices = std::vector<int>    (numHyperbolicDistRestraints, -1);
    h_torsionRestParams                   = std::vector<float3> (numTorsionRestraints, make_float3(0, 0, 0));
    h_torsionRestAtomIndices              = std::vector<int4>   (numTorsionRestraints, make_int4(-1,-1,-1,-1));
    h_torsionRestGlobalIndices            = std::vector<int>    (numTorsionRestraints, -1);
    h_distProfileRestAtomIndices          = std::vector<int2>   (numDistProfileRestraints, make_int2(-1, -1));
    h_distProfileRestDistRanges           = std::vector<float2> (numDistProfileRestraints, make_float2(0, 0));
    h_distProfileRestNumBins              = std::vector<int>    (numDistProfileRestraints, -1);
    h_distProileRestParamBounds           = std::vector<int2>   (numDistProfileRestraints, make_int2(-1, -1));
    h_distProfileRestParams               = std::vector<float4> (numDistProfileRestParams, make_float4(0, 0, 0, 0));
    h_distProfileRestScaleFactor          = std::vector<float>  (numDistProfileRestraints, 0);
    h_distProfileRestGlobalIndices        = std::vector<int>    (numDistProfileRestraints, -1);
    h_torsProfileRestAtomIndices0         = std::vector<int4>   (numTorsProfileRestraints, make_int4(-1, -1, -1, -1));
    h_torsProfileRestAtomIndices1         = std::vector<int4>   (numTorsProfileRestraints, make_int4(-1, -1, -1, -1));
    h_torsProfileRestNumBins              = std::vector<int>    (numTorsProfileRestraints, -1);
    h_torsProileRestParamBounds           = std::vector<int2>   (numTorsProfileRestraints, make_int2(-1, -1));
    h_torsProfileRestParams0              = std::vector<float4> (numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestParams1              = std::vector<float4> (numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestParams2              = std::vector<float4> (numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestParams3              = std::vector<float4> (numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestScaleFactor          = std::vector<float>  (numTorsProfileRestraints, 0);
    h_torsProfileRestGlobalIndices        = std::vector<int>    (numTorsProfileRestraints, -1);
    h_groupRestraintIndices               = std::vector<int>    (numRestraints, -1);
    h_groupBounds                         = std::vector<int2>   (numGroups, make_int2( -1, -1));
    h_groupNumActive                      = std::vector<int>    (numGroups, -1);
    h_collectionGroupIndices              = std::vector<int>    (numGroups, -1);
    h_collectionBounds                    = std::vector<int2>   (numCollections, make_int2( -1, -1));
    h_collectionNumActive                 = std::vector<int>    (numCollections, -1);
}


/**
 * Error checking helper routines
 */

void checkAtomIndex(const int numAtoms, const std::string& restType, const int atomIndex,
                const int restIndex, const int globalIndex) {
    bool bad = false;
    if (atomIndex < 0) {
        bad = true;
    }
    if (atomIndex >= numAtoms) {
        bad = true;
    }
    if (bad) {
        std::stringstream m;
        m<<"Bad index given in "<<restType<<". atomIndex is "<<atomIndex;
        m<<", globalIndex is: "<<globalIndex<<", restraint index is: "<<restIndex;
        throw OpenMMException(m.str());
    }
}


void checkForceConstant(const float forceConst, const std::string& restType,
                        const int restIndex, const int globalIndex) {
    if (forceConst < 0) {
        std::stringstream m;
        m<<"Force constant is < 0 for "<<restType<<" at globalIndex "<<globalIndex<<", restraint index "<<restIndex;
        throw OpenMMException(m.str());
    }
}


void checkDistanceRestraintRs(const float r1, const float r2, const float r3,
                              const float r4, const int restIndex, const int globalIndex) {
    std::stringstream m;
    bool bad = false;
    m<<"Distance restraint has ";

    if (r1 > r2) {
        m<<"r1 > r2. ";
        bad = true;
    } else if (r2 > r3) {
        m<<"r2 > r3. ";
        bad = true;
    } else if (r3 > r4) {
        m<<"r3 > r4. ";
        bad = true;
    }

    if (bad) {
        m<<"Restraint has index "<<restIndex<<" and globalIndex "<<globalIndex<<".";
        throw OpenMMException(m.str());
    }
}


void checkTorsionRestraintAngles(const float phi, const float deltaPhi, const int index, const int globalIndex) {
    std::stringstream m;
    bool bad = false;

    if ((phi < -180.) || (phi > 180.)) {
        m<<"Torsion restraint phi lies outside of [-180, 180]. ";
        bad = true;
    }
    if ((deltaPhi < 0) || (deltaPhi > 180)) {
        m<<"Torsion restraint deltaPhi lies outside of [0, 180]. ";
        bad = true;
    }
    if (bad) {
        m<<"Restraint has index "<<index<<" and globalIndex "<<globalIndex<<".";
        throw OpenMMException(m.str());
    }
}


void checkGroupCollectionIndices(const int num, const std::vector<int>& indices,
                                 std::vector<int>& assigned, const int index,
                                 const std::string& type1, const std::string& type2) {
    std::stringstream m;
    for(std::vector<int>::const_iterator i=indices.begin(); i!=indices.end(); ++i) {
        // make sure we're in range
        if ((*i >= num) || (*i < 0)) {
            m<<type2<<" with index "<<index<<" references "<<type1<<" outside of range[0,"<<(num-1)<<"].";
            throw OpenMMException(m.str());
        }
        // check to see if this restraint is already assigned to another group
        if (assigned[*i] != -1) {
            m<<type1<<" with index "<<(*i)<<" is assinged to more than one "<<type2<<". ";
            m<<type2<<"s are "<<assigned[*i]<<" and ";
            m<<index<<".";
            throw OpenMMException(m.str());
        }
        // otherwise mark this group as belonging to us
        else {
            assigned[*i] = index;
        }
    }
}


void checkNumActive(const std::vector<int>& indices, const int numActive, const int index, const std::string& type) {
    if ( (numActive < 0) || (numActive > indices.size()) ) {
        std::stringstream m;
        m<<type<<" with index "<<index<<" has numActive out of range [0,"<<indices.size()<<"].";
        throw OpenMMException(m.str());
    }
}


void checkAllAssigned(const std::vector<int>& assigned, const std::string& type1, const std::string& type2) {
    for (std::vector<int>::const_iterator i=assigned.begin(); i!=assigned.end(); ++i) {
        if (*i == -1) {
            std::stringstream m;
            int index = std::distance(assigned.begin(), i);
            m<<type1<<" with index "<<index<<" is not assigned to a "<<type2<<".";
            throw OpenMMException(m.str());
        }
    }
}


void CudaCalcMeldForceKernel::setupDistanceRestraints(const MeldForce& force) {
    int numAtoms = system.getNumParticles();
    std::string restType = "distance restraint";
    for (int i=0; i < numDistRestraints; ++i) {
        int atom_i, atom_j, global_index;
        float r1, r2, r3, r4, k;
        bool doing_eco;
        float eco_factor, eco_constant, eco_linear;
        int res_index1;
        int res_index2;
        force.getDistanceRestraintParams(i, atom_i, atom_j, r1, r2, r3, r4, k, doing_eco, eco_factor, eco_constant, eco_linear, res_index1, res_index2, global_index);

        checkAtomIndex(numAtoms, restType, atom_i, i, global_index);
        checkAtomIndex(numAtoms, restType, atom_j, i, global_index);
        checkForceConstant(k, restType, i, global_index);
        checkDistanceRestraintRs(r1, r2, r3, r4, i, global_index);

        h_distanceRestRParams[i] = make_float4(r1, r2, r3, r4);
        h_distanceRestKParams[i] = k;
        h_distanceRestDoingEco[i] = doing_eco;
        h_distanceRestEcoFactors[i] = eco_factor;
        h_distanceRestEcoConstants[i] = eco_constant;
        h_distanceRestEcoLinears[i] = eco_linear;
        //h_distanceRestCOValues[i] = abs((float)res_index2 - (float)res_index1); // compute the true contact order
        //cout << "res_index1:" << res_index1 << " res_index2:" << res_index2 << " CO:" << h_distanceRestCOValues[i] << "\n";

        //h_distanceRestEcoValues[i] = 1.0; // LANE: we need to have the MELD code fill this value with the eco between the two atoms of this restraint

        h_distanceRestAtomIndices[i] = make_int2(atom_i, atom_j);
        h_distanceRestResidueIndices[i] = make_int2(res_index1, res_index2);
        h_distanceRestGlobalIndices[i] = global_index;
        
    }
    
    //cout << "alpha carbons: ";
    for (int i=0; i < numResidues; ++i) {
        h_alphaCarbons[i] = force.getAlphaCarbons()[i];
        //cout << h_alphaCarbons[i] << " ";
    }
    //cout << "\n";
    //cout << "numDistRestrains: " << numDistRestraints << "\n"; 
    //cout << "Distance Restraint Sorted:";
    for (int i = 0; i < numDistRestraints; i++) {
      h_distRestSorted[i*3] = force.getDistRestSorted()[i*3];
      h_distRestSorted[i*3+1] = force.getDistRestSorted()[i*3+1];
      h_distRestSorted[i*3+2] = force.getDistRestSorted()[i*3+2];
      //cout << h_distRestSorted[i*3] << "-" << h_distRestSorted[i*3 + 1] << ":" << h_distRestSorted[i*3 + 2] << " ";
    }
    //cout << "\n";
    
    //cout << "numResidues: " << numResidues << "\n"; 
    
}


void CudaCalcMeldForceKernel::setupHyperbolicDistanceRestraints(const MeldForce& force) {
    int numAtoms = system.getNumParticles();
    std::string restType = "hyperbolic distance restraint";
    for (int i=0; i < numHyperbolicDistRestraints; ++i) {
        int atom_i, atom_j, global_index;
        float r1, r2, r3, r4, k1, k2, asymptote;
        force.getHyperbolicDistanceRestraintParams(i, atom_i, atom_j, r1, r2, r3, r4, k1, asymptote, global_index);

        checkAtomIndex(numAtoms, restType, atom_i, i, global_index);
        checkAtomIndex(numAtoms, restType, atom_j, i, global_index);
        checkForceConstant(k1, restType, i, global_index);
        checkForceConstant(asymptote, restType, i, global_index);
        checkDistanceRestraintRs(r1, r2, r3, r4, i, global_index);

        float a = 3 * (r4 - r3) * (r4 - r3);
        float b = -2 * (r4 - r3) * (r4 - r3) * (r4 - r3);
        k2 = 2.0 * asymptote / a;

        h_hyperbolicDistanceRestRParams[i] = make_float4(r1, r2, r3, r4);
        h_hyperbolicDistanceRestParams[i] = make_float4(k1, k2, a, b);
        h_hyperbolicDistanceRestAtomIndices[i] = make_int2(atom_i, atom_j);
        h_hyperbolicDistanceRestGlobalIndices[i] = global_index;
    }
}


void CudaCalcMeldForceKernel::setupTorsionRestraints(const MeldForce& force) {
    int numAtoms = system.getNumParticles();
    std::string restType = "torsion restraint";
    for (int i=0; i < numTorsionRestraints; ++i) {
        int atom_i, atom_j, atom_k, atom_l, globalIndex;
        float phi, deltaPhi, forceConstant;
        force.getTorsionRestraintParams(i, atom_i, atom_j, atom_k, atom_l, phi, deltaPhi, forceConstant, globalIndex);

        checkAtomIndex(numAtoms, restType, atom_i, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom_j, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom_k, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom_l, i, globalIndex);
        checkForceConstant(forceConstant, restType, i, globalIndex);
        checkTorsionRestraintAngles(phi, deltaPhi, i, globalIndex);

        h_torsionRestParams[i] = make_float3(phi, deltaPhi, forceConstant);
        h_torsionRestAtomIndices[i] = make_int4(atom_i, atom_j, atom_k, atom_l);
        h_torsionRestGlobalIndices[i] = globalIndex;
    }
}


void CudaCalcMeldForceKernel::setupDistProfileRestraints(const MeldForce& force) {
    int numAtoms = system.getNumParticles();
    std::string restType = "distance profile restraint";
    int currentParamIndex = 0;
    for (int i=0; i < numDistProfileRestraints; ++i) {
        int thisStart = currentParamIndex;

        int atom1, atom2, nBins, globalIndex;
        float rMin, rMax, scaleFactor;
        std::vector<double> a0, a1, a2, a3;

        force.getDistProfileRestraintParams(i, atom1, atom2, rMin, rMax, nBins,
                a0, a1, a2, a3, scaleFactor, globalIndex);

        checkAtomIndex(numAtoms, restType, atom1, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom2, i, globalIndex);
        checkForceConstant(scaleFactor, restType, i, globalIndex);

        h_distProfileRestAtomIndices[i] = make_int2(atom1, atom2);
        h_distProfileRestDistRanges[i] = make_float2(rMin, rMax);
        h_distProfileRestNumBins[i] = nBins;
        h_distProfileRestGlobalIndices[i] = globalIndex;
        h_distProfileRestScaleFactor[i] = scaleFactor;

        for (int j=0; j<nBins; ++j) {
            h_distProfileRestParams[currentParamIndex] = make_float4(
                    (float)a0[j],
                    (float)a1[j],
                    (float)a2[j],
                    (float)a3[j]);
            currentParamIndex++;
        }
        int thisEnd = currentParamIndex;
        h_distProileRestParamBounds[i] = make_int2(thisStart, thisEnd);
    }
}

void CudaCalcMeldForceKernel::setupTorsProfileRestraints(const MeldForce& force){
    int numAtoms = system.getNumParticles();
    std::string restType = "torsion profile restraint";
    int currentParamIndex = 0;
    for (int i=0; i < numTorsProfileRestraints; ++i) {
        int thisStart = currentParamIndex;

        int atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins, globalIndex;
        float scaleFactor;
        std::vector<double> a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;

        force.getTorsProfileRestraintParams(i, atom1, atom2, atom3, atom4,
                atom5, atom6, atom7, atom8, nBins,
                a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
                scaleFactor, globalIndex);

        checkAtomIndex(numAtoms, restType, atom1, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom2, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom3, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom4, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom5, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom6, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom7, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom8, i, globalIndex);
        checkForceConstant(scaleFactor, restType, i, globalIndex);

        h_torsProfileRestAtomIndices0[i] = make_int4(atom1, atom2, atom3, atom4);
        h_torsProfileRestAtomIndices1[i] = make_int4(atom5, atom6, atom7, atom8);
        h_torsProfileRestNumBins[i] = nBins;
        h_torsProfileRestGlobalIndices[i] = globalIndex;
        h_torsProfileRestScaleFactor[i] = scaleFactor;

        for (int j=0; j<nBins*nBins; ++j) {
            h_torsProfileRestParams0[currentParamIndex] = make_float4(
                    (float)a0[j],
                    (float)a1[j],
                    (float)a2[j],
                    (float)a3[j]);
            h_torsProfileRestParams1[currentParamIndex] = make_float4(
                    (float)a4[j],
                    (float)a5[j],
                    (float)a6[j],
                    (float)a7[j]);
            h_torsProfileRestParams2[currentParamIndex] = make_float4(
                    (float)a8[j],
                    (float)a9[j],
                    (float)a10[j],
                    (float)a11[j]);
            h_torsProfileRestParams3[currentParamIndex] = make_float4(
                    (float)a12[j],
                    (float)a13[j],
                    (float)a14[j],
                    (float)a15[j]);
            currentParamIndex++;
        }
        int thisEnd = currentParamIndex;
        h_torsProileRestParamBounds[i] = make_int2(thisStart, thisEnd);
    }
}

void CudaCalcMeldForceKernel::setupGroups(const MeldForce& force) {
    largestGroup = 0;
    std::vector<int> restraintAssigned(numRestraints, -1);
    int start = 0;
    int end = 0;
    for (int i=0; i<numGroups; ++i) {
        std::vector<int> indices;
        int numActive;
        force.getGroupParams(i, indices, numActive);

        checkGroupCollectionIndices(numRestraints, indices, restraintAssigned, i, "Restraint", "Group");
        checkNumActive(indices, numActive, i, "Group");

        int groupSize = indices.size();
        if (groupSize > largestGroup) {
            largestGroup = groupSize;
        }

        end = start + groupSize;
        h_groupNumActive[i] = numActive;
        h_groupBounds[i] = make_int2(start, end);

        for (int j=0; j<indices.size(); ++j) {
            h_groupRestraintIndices[start+j] = indices[j];
        }
        start = end;
    }
    checkAllAssigned(restraintAssigned, "Restraint", "Group");
}


void CudaCalcMeldForceKernel::setupCollections(const MeldForce& force) {
    largestCollection = 0;
    std::vector<int> groupAssigned(numGroups, -1);
    int start=0;
    int end=0;
    for (int i=0; i<numCollections; ++i) {
        std::vector<int> indices;
        int numActive;
        force.getCollectionParams(i, indices, numActive);

        checkGroupCollectionIndices(numGroups, indices, groupAssigned, i, "Group", "Collection");
        checkNumActive(indices, numActive, i, "Collection");

        int collectionSize = indices.size();

        if (collectionSize > largestCollection) {
            largestCollection = collectionSize;
        }

        end = start + collectionSize;
        h_collectionNumActive[i] = numActive;
        h_collectionBounds[i] = make_int2(start, end);
        for (int j=0; j<indices.size(); ++j) {
            h_collectionGroupIndices[start+j] = indices[j];
        }
        start = end;
    }
    checkAllAssigned(groupAssigned, "Group", "Collection");
}


void CudaCalcMeldForceKernel::validateAndUpload() {
    int counter;
    if (numDistRestraints > 0) {
        distanceRestRParams->upload(h_distanceRestRParams);
        distanceRestKParams->upload(h_distanceRestKParams);
        distanceRestDoingEco->upload(h_distanceRestDoingEco);
        distanceRestEcoFactors->upload(h_distanceRestEcoFactors);
        distanceRestEcoConstants->upload(h_distanceRestEcoConstants);
        distanceRestEcoLinears->upload(h_distanceRestEcoLinears);
        //distanceRestCOValues->upload(h_distanceRestCOValues);
	alphaCarbons->upload(h_alphaCarbons);
        distanceRestAtomIndices->upload(h_distanceRestAtomIndices);
        distanceRestResidueIndices->upload(h_distanceRestResidueIndices);
        distanceRestGlobalIndices->upload(h_distanceRestGlobalIndices);
    }
    
    /*
    cout << "CO values:\n";
    for (counter=0; counter < numDistRestraints; counter++) {
      cout << counter << ":" << h_distanceRestCOValues[counter] << " ";
    }
    cout << "\n";
    */
    
    if (numHyperbolicDistRestraints > 0) {
        hyperbolicDistanceRestRParams->upload(h_hyperbolicDistanceRestRParams);
        hyperbolicDistanceRestParams->upload(h_hyperbolicDistanceRestParams);
        hyperbolicDistanceRestAtomIndices->upload(h_hyperbolicDistanceRestAtomIndices);
        hyperbolicDistanceRestGlobalIndices->upload(h_hyperbolicDistanceRestGlobalIndices);
    }

    if (numTorsionRestraints > 0) {
        torsionRestParams->upload(h_torsionRestParams);
        torsionRestAtomIndices->upload(h_torsionRestAtomIndices);
        torsionRestGlobalIndices->upload(h_torsionRestGlobalIndices);
    }

    if (numDistProfileRestraints > 0) {
        distProfileRestAtomIndices->upload(h_distProfileRestAtomIndices);
        distProfileRestDistRanges->upload(h_distProfileRestDistRanges);
        distProfileRestNumBins->upload(h_distProfileRestNumBins);
        distProfileRestParamBounds->upload(h_distProileRestParamBounds);
        distProfileRestParams->upload(h_distProfileRestParams);
        distProfileRestScaleFactor->upload(h_distProfileRestScaleFactor);
        distProfileRestGlobalIndices->upload(h_distProfileRestGlobalIndices);
    }

    if (numTorsProfileRestraints > 0) {
        torsProfileRestAtomIndices0->upload(h_torsProfileRestAtomIndices0);
        torsProfileRestAtomIndices1->upload(h_torsProfileRestAtomIndices1);
        torsProfileRestNumBins->upload(h_torsProfileRestNumBins);
        torsProfileRestParamBounds->upload(h_torsProileRestParamBounds);
        torsProfileRestParams0->upload(h_torsProfileRestParams0);
        torsProfileRestParams1->upload(h_torsProfileRestParams1);
        torsProfileRestParams2->upload(h_torsProfileRestParams2);
        torsProfileRestParams3->upload(h_torsProfileRestParams3);
        torsProfileRestScaleFactor->upload(h_torsProfileRestScaleFactor);
        torsProfileRestGlobalIndices->upload(h_torsProfileRestGlobalIndices);
    }

    groupRestraintIndices->upload(h_groupRestraintIndices);
    groupBounds->upload(h_groupBounds);
    groupNumActive->upload(h_groupNumActive);
    collectionGroupIndices->upload(h_collectionGroupIndices);
    collectionBounds->upload(h_collectionBounds);
    collectionNumActive->upload(h_collectionNumActive);
}


void CudaCalcMeldForceKernel::initialize(const System& system, const MeldForce& force) {
    cu.setAsCurrent();

    allocateMemory(force);
    setupDistanceRestraints(force);
    setupHyperbolicDistanceRestraints(force);
    setupTorsionRestraints(force);
    setupDistProfileRestraints(force);
    setupTorsProfileRestraints(force);
    setupGroups(force);
    setupCollections(force);
    validateAndUpload();

    std::map<std::string, std::string> replacements;
    std::map<std::string, std::string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    replacements["MAXGROUPSIZE"] = cu.intToString(largestGroup);
    replacements["MAXCOLLECTIONSIZE"] = cu.intToString(largestCollection);

    // setup thr maximum number of groups calculated in a single block
    // want to maximize occupancy, but need to ensure that we fit
    // into shared memory
    int sharedSizeGroup = largestGroup * (sizeof(float) + sizeof(int));
    int sharedSizeThreads = 32 * sizeof(float);
    int sharedSize = std::max(sharedSizeGroup, sharedSizeThreads);
    int maxSharedMemory = 48 * 1024;
    groupsPerBlock = std::min(maxSharedMemory / sharedSize, 32);
    if (groupsPerBlock < 1) {
        throw OpenMMException("One of the groups is too large to fit into shared memory.");
    }
    replacements["GROUPSPERBLOCK"] = cu.intToString(groupsPerBlock);

    CUmodule module = cu.createModule(cu.replaceStrings(CudaMeldKernelSources::vectorOps + CudaMeldKernelSources::computeMeld, replacements), defines);
    computeDistRestKernel = cu.getKernel(module, "computeDistRest");
    computeHyperbolicDistRestKernel = cu.getKernel(module, "computeHyperbolicDistRest");
    computeTorsionRestKernel = cu.getKernel(module, "computeTorsionRest");
    computeDistProfileRestKernel = cu.getKernel(module, "computeDistProfileRest");
    computeTorsProfileRestKernel = cu.getKernel(module, "computeTorsProfileRest");
    evaluateAndActivateKernel = cu.getKernel(module, "evaluateAndActivate");
    evaluateAndActivateCollectionsKernel = cu.getKernel(module, "evaluateAndActivateCollections");
    applyGroupsKernel = cu.getKernel(module, "applyGroups");
    applyDistRestKernel = cu.getKernel(module, "applyDistRest");
    applyHyperbolicDistRestKernel = cu.getKernel(module, "applyHyperbolicDistRest");
    applyTorsionRestKernel = cu.getKernel(module, "applyTorsionRest");
    applyDistProfileRestKernel = cu.getKernel(module, "applyDistProfileRest");
    applyTorsProfileRestKernel = cu.getKernel(module, "applyTorsProfileRest");
    computeContactsKernel = cu.getKernel(module, "computeContacts");
    computeEdgeListKernel = cu.getKernel(module, "computeEdgeList");
    dijkstra_initializeKernel = cu.getKernel(module, "dijkstra_initialize");
    dijkstra_save_old_vectorsKernel = cu.getKernel(module, "dijkstra_save_old_vectors");
    dijkstra_settle_and_updateKernel = cu.getKernel(module, "dijkstra_settle_and_update");
    dijkstra_log_reduceKernel = cu.getKernel(module, "dijkstra_log_reduce");
    assignRestEcoKernel = cu.getKernel(module, "assignRestEco");
    test_get_alpha_carbon_posqKernel = cu.getKernel(module, "test_get_alpha_carbon_posq");
}


void CudaCalcMeldForceKernel::copyParametersToContext(ContextImpl& context, const MeldForce& force) {
    cu.setAsCurrent();

    setupDistanceRestraints(force);
    setupHyperbolicDistanceRestraints(force);
    setupTorsionRestraints(force);
    setupDistProfileRestraints(force);
    setupTorsProfileRestraints(force);
    setupGroups(force);
    setupCollections(force);
    validateAndUpload();

    // Mark that the current reordering may be invalid.
    cu.invalidateMolecules();
}

/*
 * This function handles the calculation of the eco values. First a graph of contacts is created, then shortest paths are calculated.
 * The eco values are stored in distanceRestEcoValues on the device.
 */
void CudaCalcMeldForceKernel::calcEcoValues() {
  int counter;
  int counter2;
  int src = -1; // so the pathfinding algorithm is run with the first restraint
  int dest = 0;
  int num_explored = 1;
  struct timeval endtime;
  long int starttime;
  long int timediff;
  long int timesum = 0;
  long int timecount = 0;
  
  gettimeofday(&endtime, NULL);
    void* contactsArgs[] = {
	&cu.getPosq().getDevicePointer(),
	&distanceRestAtomIndices->getDevicePointer(),
	&numResidues,
	&ecoCutoff,
	&distanceRestContacts->getDevicePointer(),
	&alphaCarbons->getDevicePointer()};

    cu.executeKernel(computeContactsKernel, contactsArgs, numResidues*numResidues);
  
    void* edgeListArgs[] = {
	&distanceRestContacts->getDevicePointer(),
	&distanceRestEdgeCounts->getDevicePointer(),
	&numResidues};

    cu.executeKernel(computeEdgeListKernel, edgeListArgs, numResidues);
  
  
  // TESTING PURPOSES: delete for final version
  // I want to pull over distanceRestContacts and distanceRestEdgeCounts
  //distanceRestContacts->download(h_distanceRestContacts);
  //distanceRestEdgeCounts->download(h_distanceRestEdgeCounts);
  
  /*cout << "Edge list:\n";
  for (counter = 0; counter < numResidues; counter++) {
    cout << "Node number: " << counter << " edge count:" << h_distanceRestEdgeCounts[counter] << "\n";
    for (counter2 = 0; counter2 < h_distanceRestEdgeCounts[counter]; counter2++) {
      cout << h_distanceRestContacts[counter * numResidues + counter2] << " ";
    }
    cout << "\n";
  } */
  
  
  // for each of the restraints, do the following...
  counter = 0;
  
  
  
  // Define all the arguments by reference that will be passed to the CUDA kernels
    void* dijkstra_initializeArgs[] = {
  &dijkstra_unexplored->getDevicePointer(),
  &dijkstra_frontier->getDevicePointer(),
  &dijkstra_distance->getDevicePointer(),
  &dijkstra_n_explored->getDevicePointer(),
  &src,
  &INF,
  &numResidues};
  
    void* dijkstra_save_oldArgs[] = {
  &dijkstra_unexplored->getDevicePointer(),
  &dijkstra_unexplored_old->getDevicePointer(),
  &dijkstra_frontier->getDevicePointer(),
  &dijkstra_frontier_old->getDevicePointer(),
  &numResidues};
  
    void* dijkstra_settle_and_updateArgs[] = {
  &dijkstra_unexplored->getDevicePointer(),
  &dijkstra_unexplored_old->getDevicePointer(),
  &dijkstra_frontier->getDevicePointer(),
  &dijkstra_frontier_old->getDevicePointer(),
  &dijkstra_distance->getDevicePointer(),
  &distanceRestEdgeCounts->getDevicePointer(),
  &distanceRestContacts->getDevicePointer(),
  &dijkstra_n_explored->getDevicePointer(),
  &counter2,
  &numResidues};
  
    void* dijkstra_log_reduceArgs[] = {
  &numResidues,
  &dijkstra_n_explored->getDevicePointer(),
  &dijkstra_total->getDevicePointer()};
  
    void* assignRestEcoArgs[] = {
  &src,
  &distanceRestResidueIndices->getDevicePointer(),
  &dijkstra_distance->getDevicePointer(),
  &distanceRestEcoValues->getDevicePointer()
    };
  
  int rest_index;
  
  /*
  starttime = endtime.tv_usec;
  gettimeofday(&endtime, NULL);
  timediff = (long int)(endtime.tv_usec) - starttime;
  cout << "Time to compute input arguments: " << timediff << "\n";
  gettimeofday(&endtime, NULL);*/
  
  for (counter = 0; counter < numDistRestraints; counter++) {
    
    rest_index = h_distRestSorted[counter*3 + 2]; // the index of the distance restraint that we are computing ECO for
    
    /*cout << "h_distRestSorted:\n";
    for (counter2 = 0; counter2 < numDistRestraints; counter2++) {
      cout << h_distRestSorted[counter2*3] << "-" << h_distRestSorted[counter2*3 + 1] << "-" << h_distRestSorted[counter2*3 + 2] << " ";
    }
    cout << "\n"; */
    //cout << "counter: " << counter << " rest_index: " << rest_index << " h_distanceRestResidueIndices[rest_index].x: " << h_distanceRestResidueIndices[rest_index].x << "\n";
    if ((h_distanceRestResidueIndices[rest_index].x > src) && (h_distanceRestDoingEco[rest_index] == 1)) { // if we have a new source for this restraint (since they're sorted)
      src = h_distanceRestResidueIndices[rest_index].x; // then update this source index and rerun Dijkstra
      //cout << "now on source: " << src << ". Rerunning Dijkstra's\n";
      
      cu.executeKernel(dijkstra_initializeKernel, dijkstra_initializeArgs, numResidues); // initialize Dijkstra variables
      counter2 = 0;
      num_explored = 1; // the source at least has been explored
      while ((counter2 <= 5) && (num_explored < numResidues)) { // (counter2 <= numResidues-1)
        cu.executeKernel(dijkstra_save_old_vectorsKernel, dijkstra_save_oldArgs, numResidues); // save the old arrays from the past step
        cu.executeKernel(dijkstra_settle_and_updateKernel, dijkstra_settle_and_updateArgs, numResidues); // update exploration values
        //cu.executeKernel(dijkstra_log_reduceKernel, dijkstra_log_reduceArgs, numResidues); // efficiently determine how many were explored
        //cout << "mark20\n";
        //dijkstra_total->download(h_dijkstra_total); // pull the number of explored residues to the CPU
        //cout << "mark21\n";
        //num_explored += h_dijkstra_total[0]; // increment the total number explored
        //cout << "num_explored:" << num_explored << "\n";
        counter2++;
      }
      
      //dijkstra_distance->download(h_dijkstra_distance); // NOTE: Remove???
      cu.executeKernel(assignRestEcoKernel, assignRestEcoArgs, numDistRestraints); // give each distance restraint its ECO value
      /*cout << "Distance vector from src: " << src << "\n";
      for (counter2 = 0; counter2 < numResidues; counter2++) {
        cout << h_dijkstra_distance[counter2] << " ";
      }
      cout << "\n"; */
    }
    //dest = h_distanceRestResidueIndices[rest_index].y; // NOTE: remove?
    //cout << "eco for src: " << src << " dest: " << dest << " dist:" << h_dijkstra_distance[dest] << "\n";
  }
  //cout << "End\n"; 
  
  //cout << "Average time to settle: " << timevar / timecount << "\n";
  
  /*
  starttime = endtime.tv_usec;
  gettimeofday(&endtime, NULL);
  timediff = (long int)(endtime.tv_usec) - starttime;
  cout << "Time to do Dijkstra for all restraints: " << timediff << "\n";
  */
  
}

void CudaCalcMeldForceKernel::testEverythingEco() {
  int counter, counter2, contact_ptr, order;
  float err_tol = 0.0001;
  float dist_sq;
  float x, y, z;
  int src = 0;
  int num_explored = 1;
  int edge_index, head;
  
  // We need to run a series of tests to make sure that everything is behaving like we expect
  void* test_get_alpha_carbon_posqArgs[] = {
    &cu.getPosq().getDevicePointer(),
    //&alphaCarbonPosq->getDevicePointer(),
    &alphaCarbons->getDevicePointer(),
    &numResidues
  };
  //cu.executeKernel(test_get_alpha_carbon_posqKernel, test_get_alpha_carbon_posqArgs, numResidues);
  //cout << "mark0\n";
  //alphaCarbonPosq->download(h_alphaCarbonPosq);
  /*cout << "Alpha Carbon x,y,z:\n";
  for (counter = 0; counter < numResidues; counter++) {
    cout << h_alphaCarbonPosq[counter*3] << "," << h_alphaCarbonPosq[counter*3 + 1] << "," << h_alphaCarbonPosq[counter*3 + 2] << " ";
  }
  cout << "\n";  */
  distanceRestContacts->download(h_distanceRestContacts);
  
  /*
  for (counter = 0; counter < numResidues; counter++) {
    contact_ptr = 0;
    for (counter2 = 0; counter2 < numResidues; counter2++) {
      x = h_alphaCarbonPosq[counter*3] - h_alphaCarbonPosq[counter2*3];
      y = h_alphaCarbonPosq[counter*3 + 1] - h_alphaCarbonPosq[counter2*3 + 1];
      z = h_alphaCarbonPosq[counter*3 + 2] - h_alphaCarbonPosq[counter2*3 + 2];
      dist_sq = x*x + y*y + z*z;
      if ( h_distanceRestContacts[numResidues * counter + contact_ptr] == counter2 ) { // then we've hit an edge
        //cout << "Edge between node: " << counter << " and node: " << counter2 << "\n";
        if (dist_sq > ecoCutoff*ecoCutoff && (counter != counter2 - 1 || counter != counter2 + 1)) { // so if the actual distance is greater than expected, then something's wrong
          cout << "ERROR: contact map problem: counter: " << counter << " counter2: " << counter2 << " contact predicted to exist yet distance squared is: " << dist_sq << "\n";
        }
        contact_ptr++; // increment this pointer
      } else { // no contact predicted
        if (dist_sq < ecoCutoff*ecoCutoff && counter != counter2) { // so if the actual distance is less than cutoff, then something's wrong
          cout << "ERROR: contact map problem: counter: " << counter << " counter2: " << counter2 << " contact predicted not to exist yet distance squared is: " << dist_sq << "\n";
        }
      }
    }
  }
  */
  
  // Now test Dijkstra's algorithm results
  for (counter = 0; counter < numResidues; counter++) { // first, initialize the arrays
    h_dijkstra_unexplored[counter] = true;
    h_dijkstra_frontier[counter] = false;
    h_dijkstra_distance[counter] = INF;
    h_dijkstra_n_explored[counter] = 0;
    if (counter == src) {
      h_dijkstra_unexplored[counter] = false;
      h_dijkstra_frontier[counter] = true;
      h_dijkstra_distance[counter] = 0;
    }
  }
  
  
  distanceRestEdgeCounts->download(h_distanceRestEdgeCounts);
  
  counter2 = 0;
  num_explored = 1;
  while ((counter2 <= numResidues + 2) && (num_explored < numResidues)) {
    for (counter = 0; counter < numResidues; counter++) { // save the old arrays
      h_dijkstra_unexplored_old[counter] = h_dijkstra_unexplored[counter];
      h_dijkstra_frontier_old[counter] = h_dijkstra_frontier[counter];
    }
    
    for (counter = 0; counter < numResidues; counter++) { // settle and update
      h_dijkstra_n_explored[counter] = 0;
      if (h_dijkstra_unexplored_old[counter] == true) {
        for (contact_ptr = 0; contact_ptr < h_distanceRestEdgeCounts[counter]; contact_ptr++) {
          edge_index = (numResidues * counter) + contact_ptr;
          head = h_distanceRestContacts[edge_index];  // the index of the node that is leading to this one
          //cout << "edge from node " << counter << " to " << head << "\n";
          if (h_dijkstra_frontier_old[head] == true) { // if the head node is on the frontier, the we need to change our explored status
            h_dijkstra_frontier[counter] = true; // then add myself to the frontier
            h_dijkstra_unexplored[counter] = false; // remove myself from the unexplored
            h_dijkstra_distance[counter] = counter2 + 1; // the number of iterations we've needed to find me is the distance
            num_explored++;
            break;
          }
        }
      }
    }
    counter2++;
  }
  
  // Now rerun the GPU pathfinding algorithm
   void* dijkstra_initializeArgs[] = {
  &dijkstra_unexplored->getDevicePointer(),
  &dijkstra_frontier->getDevicePointer(),
  &dijkstra_distance->getDevicePointer(),
  &dijkstra_n_explored->getDevicePointer(),
  &src,
  &INF,
  &numResidues};
  
    void* dijkstra_save_oldArgs[] = {
  &dijkstra_unexplored->getDevicePointer(),
  &dijkstra_unexplored_old->getDevicePointer(),
  &dijkstra_frontier->getDevicePointer(),
  &dijkstra_frontier_old->getDevicePointer(),
  &numResidues};
  
    void* dijkstra_settle_and_updateArgs[] = {
  &dijkstra_unexplored->getDevicePointer(),
  &dijkstra_unexplored_old->getDevicePointer(),
  &dijkstra_frontier->getDevicePointer(),
  &dijkstra_frontier_old->getDevicePointer(),
  &dijkstra_distance->getDevicePointer(),
  &distanceRestEdgeCounts->getDevicePointer(),
  &distanceRestContacts->getDevicePointer(),
  &dijkstra_n_explored->getDevicePointer(),
  &counter2,
  &numResidues};
  
    void* dijkstra_log_reduceArgs[] = {
  &numResidues,
  &dijkstra_n_explored->getDevicePointer(),
  &dijkstra_total->getDevicePointer()};
  cu.executeKernel(dijkstra_initializeKernel, dijkstra_initializeArgs, numResidues);
  counter2 = 0;
  num_explored = 1;
  
  while ((counter2 <= numResidues + 2) && (num_explored < numResidues)) {
    cu.executeKernel(dijkstra_save_old_vectorsKernel, dijkstra_save_oldArgs, numResidues);
    cu.executeKernel(dijkstra_settle_and_updateKernel, dijkstra_settle_and_updateArgs, numResidues);
    cu.executeKernel(dijkstra_log_reduceKernel, dijkstra_log_reduceArgs, numResidues);
    dijkstra_total->download(h_dijkstra_total); // SLOW!!!! There is a better way to do this...
    num_explored += h_dijkstra_total[0];
    counter2++;
  }
  
  // by this point, the graphs should be explored
  dijkstra_distance->download(h_dijkstra_distance2);
  for (counter = 0; counter < numResidues; counter++) {
    if (h_dijkstra_distance[counter] != h_dijkstra_distance2[counter] ) { // if these two distances are not equal, then one of the pathfinding algorithms is broken
      cout << "ERROR: pathfinding algorithm discrepancy. Src: 0. Dest: " << counter << " (CPU): " << h_dijkstra_distance[counter] << " (GPU): " << h_dijkstra_distance2[counter] << "\n";
    }
  }
  
}

double CudaCalcMeldForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // compute the forces and energies
    int oldtime = timevar; // save the old timestamp
    //struct timeval newtime;
    //gettimeofday(&newtime, NULL);
    //timevar = (long int)(newtime.tv_usec);
    
    //cout << "TIME ELAPSED: " << timevar - oldtime << "\n";
    int counter;
    if (numDistRestraints > 0) {
        calcEcoValues(); // calculate the graph that will be used in the ECO calcs
        
        distanceRestEcoValues->download(h_distanceRestEcoValues);
        cout << "ECO values per restraint (after): ";
        for (counter = 0; counter < numDistRestraints; counter++) {
          cout << h_distanceRestResidueIndices[counter].x << "-" << h_distanceRestResidueIndices[counter].y << ":" << h_distanceRestEcoValues[counter] << " ";
        }
        cout << "\n";
        
        //testEverythingEco(); // comment out this line in the final production version
        void* distanceArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &distanceRestAtomIndices->getDevicePointer(),
            //&distanceRestResidueIndices->getDevicePointer(),
            &distanceRestRParams->getDevicePointer(),
            &distanceRestKParams->getDevicePointer(),
            &distanceRestDoingEco->getDevicePointer(), 
            &distanceRestEcoFactors->getDevicePointer(),
            &distanceRestEcoConstants->getDevicePointer(),
            &distanceRestEcoLinears->getDevicePointer(),
            &distanceRestEcoValues->getDevicePointer(),
            //&distanceRestCOValues->getDevicePointer(),
            &distanceRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            //&nonECOrestraintEnergies->getDevicePointer(),
            &distanceRestForces->getDevicePointer(),
            &numDistRestraints}; // this is getting the reference pointer for each of these arrays
        cu.executeKernel(computeDistRestKernel, distanceArgs, numDistRestraints);
    }
    
    /*
    distanceRestEcoValues->download(h_distanceRestEcoValues);
    //distanceRestCOValues->download(h_distanceRestCOValues);
    
    cout << "ECO values per restraint: ";
    for (counter = 0; counter < numDistRestraints; counter++) {
      cout << h_distanceRestResidueIndices[counter].x << "-" << h_distanceRestResidueIndices[counter].y << ":" << h_distanceRestEcoValues[counter] << " ";
    }
    cout << "\n";
    /*
    restraintEnergies->download(h_restraintEnergies);
    nonECOrestraintEnergies->download(h_restraintNonEcoEnergies);
    
    cout << "ECO unmodified Energy per restraint: ";
    for (counter = 0; counter < numRestraints; counter++) {
      cout << counter << " modified: " << h_restraintEnergies[counter] << " original: " << h_restraintNonEcoEnergies[counter] << " ";
    }
    cout << "\n";
    */
    
    if (numHyperbolicDistRestraints > 0) {
        void* hyperbolicDistanceArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &hyperbolicDistanceRestAtomIndices->getDevicePointer(),
            &hyperbolicDistanceRestRParams->getDevicePointer(),
            &hyperbolicDistanceRestParams->getDevicePointer(),
            &hyperbolicDistanceRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &hyperbolicDistanceRestForces->getDevicePointer(),
            &numHyperbolicDistRestraints};
        cu.executeKernel(computeHyperbolicDistRestKernel, hyperbolicDistanceArgs, numHyperbolicDistRestraints);
    }

    if (numTorsionRestraints > 0) {
        void* torsionArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &torsionRestAtomIndices->getDevicePointer(),
            &torsionRestParams->getDevicePointer(),
            &torsionRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &torsionRestForces->getDevicePointer(),
            &numTorsionRestraints};
        cu.executeKernel(computeTorsionRestKernel, torsionArgs, numTorsionRestraints);
    }

    if (numDistProfileRestraints > 0) {
        void * distProfileArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &distProfileRestAtomIndices->getDevicePointer(),
            &distProfileRestDistRanges->getDevicePointer(),
            &distProfileRestNumBins->getDevicePointer(),
            &distProfileRestParams->getDevicePointer(),
            &distProfileRestParamBounds->getDevicePointer(),
            &distProfileRestScaleFactor->getDevicePointer(),
            &distProfileRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &distProfileRestForces->getDevicePointer(),
            &numDistProfileRestraints };
        cu.executeKernel(computeDistProfileRestKernel, distProfileArgs, numDistProfileRestraints);
    }

    if (numTorsProfileRestraints > 0) {
        void * torsProfileArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &torsProfileRestAtomIndices0->getDevicePointer(),
            &torsProfileRestAtomIndices1->getDevicePointer(),
            &torsProfileRestNumBins->getDevicePointer(),
            &torsProfileRestParams0->getDevicePointer(),
            &torsProfileRestParams1->getDevicePointer(),
            &torsProfileRestParams2->getDevicePointer(),
            &torsProfileRestParams3->getDevicePointer(),
            &torsProfileRestParamBounds->getDevicePointer(),
            &torsProfileRestScaleFactor->getDevicePointer(),
            &torsProfileRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &torsProfileRestForces->getDevicePointer(),
            &numTorsProfileRestraints };
        cu.executeKernel(computeTorsProfileRestKernel, torsProfileArgs, numTorsProfileRestraints);
    }

    // now evaluate and active restraints based on groups
    int sharedSizeGroup = largestGroup * (sizeof(float) + sizeof(int));
    int sharedSizeThreads = 32 * sizeof(float);
    int sharedSize = std::max(sharedSizeGroup, sharedSizeThreads);

    void* groupArgs[] = {
        &numGroups,
        &groupNumActive->getDevicePointer(),
        &groupBounds->getDevicePointer(),
        &groupRestraintIndices->getDevicePointer(),
        &groupRestraintIndicesTemp->getDevicePointer(),
        &restraintEnergies->getDevicePointer(),
        &restraintActive->getDevicePointer(),
        &groupEnergies->getDevicePointer()};
    cu.executeKernel(evaluateAndActivateKernel, groupArgs, 32 * numGroups, groupsPerBlock * 32, groupsPerBlock * sharedSize);

    // the kernel will need to be modified if this value is changed
    const int threadsPerCollection = 1024;
    int sharedSizeCollectionEnergies = largestCollection * sizeof(float);
    int sharedSizeCollectionMinMaxBuffer = threadsPerCollection * 2 * sizeof(float);
    int sharedSizeCollectionBinCounts = threadsPerCollection * sizeof(int);
    int sharedSizeCollectionBestBin = sizeof(int);
    int sharedSizeCollection = sharedSizeCollectionEnergies + sharedSizeCollectionMinMaxBuffer +
        sharedSizeCollectionBinCounts + sharedSizeCollectionBestBin;
    // now evaluate and activate groups based on collections
    void* collArgs[] = {
        &numCollections,
        &collectionNumActive->getDevicePointer(),
        &collectionBounds->getDevicePointer(),
        &collectionGroupIndices->getDevicePointer(),
        &groupEnergies->getDevicePointer(),
        &groupActive->getDevicePointer()};
    cu.executeKernel(evaluateAndActivateCollectionsKernel, collArgs, threadsPerCollection*numCollections, threadsPerCollection, sharedSizeCollection);

    // Now set the restraints active based on if the groups are active
    void* applyGroupsArgs[] = {
        &groupActive->getDevicePointer(),
        &restraintActive->getDevicePointer(),
        &groupBounds->getDevicePointer(),
        &numGroups};
    cu.executeKernel(applyGroupsKernel, applyGroupsArgs, 32*numGroups, 32);

    // Now apply the forces and energies if the restraints are active
    if (numDistRestraints > 0) {
        void* applyDistRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &distanceRestAtomIndices->getDevicePointer(),
            &distanceRestGlobalIndices->getDevicePointer(),
            &distanceRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            //&nonECOrestraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numDistRestraints};
        cu.executeKernel(applyDistRestKernel, applyDistRestArgs, numDistRestraints);
    }

    if (numHyperbolicDistRestraints > 0) {
        void* applyHyperbolicDistRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &hyperbolicDistanceRestAtomIndices->getDevicePointer(),
            &hyperbolicDistanceRestGlobalIndices->getDevicePointer(),
            &hyperbolicDistanceRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numHyperbolicDistRestraints};
        cu.executeKernel(applyHyperbolicDistRestKernel, applyHyperbolicDistRestArgs, numHyperbolicDistRestraints);
    }

    if (numTorsionRestraints > 0) {
        void* applyTorsionRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &torsionRestAtomIndices->getDevicePointer(),
            &torsionRestGlobalIndices->getDevicePointer(),
            &torsionRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numTorsionRestraints};
        cu.executeKernel(applyTorsionRestKernel, applyTorsionRestArgs, numTorsionRestraints);
    }

    if (numDistProfileRestraints > 0) {
        void *applyDistProfileRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &distProfileRestAtomIndices->getDevicePointer(),
            &distProfileRestGlobalIndices->getDevicePointer(),
            &distProfileRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numDistProfileRestraints
        };
        cu.executeKernel(applyDistProfileRestKernel, applyDistProfileRestArgs, numDistProfileRestraints);
    }

    if (numTorsProfileRestraints > 0) {
        void *applyTorsProfileRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &torsProfileRestAtomIndices0->getDevicePointer(),
            &torsProfileRestAtomIndices1->getDevicePointer(),
            &torsProfileRestGlobalIndices->getDevicePointer(),
            &torsProfileRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numTorsProfileRestraints
        };
        cu.executeKernel(applyTorsProfileRestKernel, applyTorsProfileRestArgs, numTorsProfileRestraints);
    }

    return 0.0;
}


/*
 * RDC Stuff
 */

CudaCalcRdcForceKernel::CudaCalcRdcForceKernel(std::string name, const Platform& platform,
        CudaContext& cu, const System& system) :
    CalcRdcForceKernel(name, platform), cu(cu), system(system) {

    if (cu.getUseDoublePrecision()) {
        cout << "***\n";
        cout << "*** RdcForce does not support double precision.\n";
        cout << "***" << endl;
        throw OpenMMException("RdcForce does not support double precision");
    }

    int numExperiments = 0;
    int numRdcRestraints = 0;

    r = NULL;
    atomExptIndices = NULL;
    lhs = NULL;
    rhs = NULL;
    S = NULL;
    kappa = NULL;
    tolerance = NULL;
    force_const = NULL;
    weight = NULL;
}

CudaCalcRdcForceKernel::~CudaCalcRdcForceKernel() {
    cu.setAsCurrent();
    delete r;
    delete atomExptIndices;
    delete lhs;
    delete rhs;
    delete S;
    delete kappa;
    delete tolerance;
    delete force_const;
    delete weight;
}

void CudaCalcRdcForceKernel::initialize(const System& system, const RdcForce& force) {
    cu.setAsCurrent();

    allocateMemory(force);
    setupRdcRestraints(force);
    validateAndUpload();

    std::map<std::string, std::string> replacements;
    std::map<std::string, std::string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    CUmodule module = cu.createModule(cu.replaceStrings(CudaMeldKernelSources::vectorOps + CudaMeldKernelSources::computeRdc, replacements), defines);
    computeRdcPhase1 = cu.getKernel(module, "computeRdcPhase1");
    computeRdcPhase3 = cu.getKernel(module, "computeRdcPhase3");
}

double CudaCalcRdcForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // TODO: the weights are currently not used

    // Phase 1
    // compute the lhs on the gpu
    void* computePhase1Args[] = {
        &numRdcRestraints,
        &cu.getPosq().getDevicePointer(),
        &atomExptIndices->getDevicePointer(),
        &kappa->getDevicePointer(),
        &r->getDevicePointer(),
        &lhs->getDevicePointer()};
    cu.executeKernel(computeRdcPhase1, computePhase1Args, numRdcRestraints);

    // Phase 2
    // download the lhs, compute S on CPU, upload S back to GPU
    computeRdcPhase2();

    // Phase 3
    // compute the energies and forces on the GPU
    void* computePhase3Args[] = {
        &numRdcRestraints,
        &cu.getPosq().getDevicePointer(),
        &atomExptIndices->getDevicePointer(),
        &kappa->getDevicePointer(),
        &S->getDevicePointer(),
        &rhs->getDevicePointer(),
        &tolerance->getDevicePointer(),
        &force_const->getDevicePointer(),
        &r->getDevicePointer(),
        &lhs->getDevicePointer(),
        &cu.getForce().getDevicePointer(),
        &cu.getEnergyBuffer().getDevicePointer()};
    cu.executeKernel(computeRdcPhase3, computePhase3Args, numRdcRestraints);
    return 0.;
}

void CudaCalcRdcForceKernel::computeRdcPhase2() {
    // Download the lhs from the gpu
    lhs->download(h_lhs);

    // loop over the experiments
    for(int i=0; i<numExperiments; ++i) {
        // get the indices for things in this experiment
        int start = h_experimentBounds[i].x;
        int end = h_experimentBounds[i].y;
        int len = end - start;

        // create wrappers for the correct parts of lhs and rhs
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > lhsWrap(&h_lhs[5 * start], len, 5);
        Eigen::Map<Eigen::VectorXf> rhsWrap(&h_rhs[start], len);
        Eigen::Map<Eigen::VectorXf> SWrap(&h_S[5 * i], 5);


        // solve for S
        SWrap = lhsWrap.jacobiSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(rhsWrap);
    }

    // upload S back up to the gpu
    S->upload(h_S);
}

void CudaCalcRdcForceKernel::copyParametersToContext(ContextImpl& context, const RdcForce& force) {
    cu.setAsCurrent();
    setupRdcRestraints(force);
    validateAndUpload();
    cu.invalidateMolecules();
}

void CudaCalcRdcForceKernel::allocateMemory(const RdcForce& force) {
    numExperiments = force.getNumExperiments();
    numRdcRestraints = force.getNumTotalRestraints();

    /*
     * Allocate device memory
     */
    r = CudaArray::create<float4> (cu, numRdcRestraints, "r");
    atomExptIndices = CudaArray::create<int3> (cu, numRdcRestraints, "atomExptIndices");
    lhs = CudaArray::create<float> (cu, 5 * numRdcRestraints, "lhs");
    rhs = CudaArray::create<float> (cu, numRdcRestraints, "rhs");
    kappa = CudaArray::create<float> (cu, numRdcRestraints, "kappa");
    tolerance = CudaArray::create<float> (cu, numRdcRestraints, "tolerance");
    force_const = CudaArray::create<float> (cu, numRdcRestraints, "force_const");
    weight = CudaArray::create<float> (cu, numRdcRestraints, "weight");
    S = CudaArray::create<float> (cu, 5 * numExperiments, "S");

    /**
     * Allocate host memory
     */
    h_atomExptIndices = std::vector<int3> (numRdcRestraints, make_int3(0, 0, 0));
    h_lhs = std::vector<float> (5 * numRdcRestraints, 0.);
    h_rhs = std::vector<float> (numRdcRestraints, 0.);
    h_kappa = std::vector<float> (numRdcRestraints, 0.);
    h_tolerance = std::vector<float> (numRdcRestraints, 0.);
    h_force_const = std::vector<float> (numRdcRestraints, 0.);
    h_weight = std::vector<float> (numRdcRestraints, 0.);
    h_experimentBounds = std::vector<int2> (numExperiments, make_int2(0, 0));
    h_S = std::vector<float> (5 * numExperiments, 0.);
}

void CudaCalcRdcForceKernel::setupRdcRestraints(const RdcForce& force) {
    int currentIndex = 0;
    // loop over the experiments
    for(int expIndex = 0; expIndex < numExperiments; ++expIndex) {
        int experimentStart = currentIndex;
        std::vector<int> restraintsInExperiment;
        force.getExperimentInfo(expIndex, restraintsInExperiment);

        // loop over the restraints
        for(int withinExpIndex = 0; withinExpIndex < force.getNumRestraints(expIndex); ++withinExpIndex) {
            int currentRestraint = restraintsInExperiment[withinExpIndex];
            int atom1, atom2, globalIndex;
            float kappa, dobs, tolerance, force_const, weight;

            force.getRdcRestraintInfo(currentRestraint, atom1, atom2, kappa, dobs, tolerance,
                    force_const, weight, globalIndex);

            h_atomExptIndices[currentIndex] = make_int3(atom1, atom2, expIndex);
            h_kappa[currentIndex] = kappa;
            h_force_const[currentIndex] = force_const;
            h_weight[currentIndex] = weight;
            h_rhs[currentIndex] = dobs;
            h_tolerance[currentIndex] = tolerance;

            currentIndex++;
        }
        int experimentEnd = currentIndex;
        h_experimentBounds[expIndex] = make_int2(experimentStart, experimentEnd);
    }
}

void CudaCalcRdcForceKernel::validateAndUpload() {
    // todo need to do better validation
    atomExptIndices->upload(h_atomExptIndices);
    lhs->upload(h_lhs);
    rhs->upload(h_rhs);
    tolerance->upload(h_tolerance);
    force_const->upload(h_force_const);
    weight->upload(h_weight);
    kappa->upload(h_kappa);
}
