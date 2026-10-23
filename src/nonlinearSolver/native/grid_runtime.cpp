// 逐标量 AVC 调度器：维护依赖、有效位和回退，不包含用户方程或求解限步。
#include <complex>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstddef>

// C ABI 在两种平台保持相同布局；导出和调用约定由平台宏隔离。
#if defined(_WIN32)
#define GRID_EXPORT __declspec(dllexport)
#define GRID_CALL __cdecl
#else
#define GRID_EXPORT __attribute__((visibility("default")))
#define GRID_CALL
#endif
using C=std::complex<double>;
struct Parent {int node,component,width; const int64_t* map;};
struct Node {int opcode,size,components,iscomplex,parent_count; void* value; Parent* parents; const double* weights; const int64_t* selector;};
struct Edge {int node,point;};
struct Target {int node,component,point;};
struct Saved {int node,point;uint8_t valid;size_t start;};
using Callback=int(GRID_CALL*)(int,int,const int*);
struct Program {
  Node* nodes;int count;const int64_t* offsets;const Edge* edges;std::vector<std::vector<uint8_t>> valid;uint64_t points=0,batches=0;
  std::vector<int64_t> starts,childOffsets;std::vector<Edge> children;
  uint32_t undoEpoch=0;bool inUndo=false;
  uint64_t validityChecks=0,invalidated=0,requested=0,savedPoints=0;
  std::vector<std::vector<int>> pending;
  std::vector<int> active;std::vector<uint8_t> queued;
  std::vector<std::vector<uint32_t>> undoMarks;
  std::vector<std::vector<uint8_t>> live;
  std::vector<Saved> journal;std::vector<C> oldValues;
  std::vector<int> oldActive;std::vector<std::vector<int>> oldPending;
  Program(Node*n,int count,const int64_t*off,const Edge*e):nodes(n),count(count),offsets(off),edges(e){
    pending.resize(count);queued.resize(count);
    for(int i=0;i<count;++i){valid.emplace_back(n[i].size,0);undoMarks.emplace_back(n[i].size,0);live.emplace_back(n[i].size,1);}}
  void buildDirect(){
    starts.clear();starts.push_back(0);for(int i=0;i<count;++i)starts.push_back(starts.back()+nodes[i].size);
    childOffsets.assign(starts.back()+1,0);
    auto visit=[&](auto emit){for(int i=0;i<count;++i){Node&n=nodes[i];if(n.opcode==-1)continue;
      for(int j=0;j<n.size;++j)if(live[i][j])for(int k=0;k<n.parent_count;++k){
        if(n.opcode==23 && k!=n.selector[j])continue;Parent&p=n.parents[k];
        for(int q=0;q<p.width;++q){if(n.opcode==22 && n.weights[j*p.width+q]==0)continue;
          emit(starts[p.node]+p.map[j*p.width+q],Edge{i,j});}}}};
    visit([&](int64_t from,Edge e){++childOffsets[from+1];});
    for(size_t i=1;i<childOffsets.size();++i)childOffsets[i]+=childOffsets[i-1];
    children.resize(childOffsets.back());auto cursor=childOffsets;
    visit([&](int64_t from,Edge e){children[cursor[from]++]=e;});
  }
  C get(int node,int component,int index){Node&n=nodes[node];int i=component*n.size+index;return n.iscomplex?((C*)n.value)[i]:C(((double*)n.value)[i],0);}
  void compileTargets(int nt,const Target*targets){
    // 方程汇点固定时，编译逐点祖先闭包；批计算器的无用点也不进入 AVC。
    for(auto&v:live)std::fill(v.begin(),v.end(),0);std::vector<Edge> work;
    for(int i=0;i<nt;++i)work.push_back({targets[i].node,targets[i].point});
    while(!work.empty()){
      Edge e=work.back();work.pop_back();if(live[e.node][e.point])continue;live[e.node][e.point]=1;
      Node&n=nodes[e.node];if(n.opcode==-1)continue;
      for(int k=0;k<n.parent_count;++k){if(n.opcode==23 && k!=n.selector[e.point])continue;Parent&p=n.parents[k];
        for(int q=0;q<p.width;++q){if(n.opcode==22 && n.weights[e.point*p.width+q]==0)continue;
          work.push_back({p.node,(int)p.map[e.point*p.width+q]});}}
    }
    buildDirect();
  }
  void set(int node,int component,int index,C value){Node&n=nodes[node];int i=component*n.size+index;if(n.iscomplex)((C*)n.value)[i]=value;else((double*)n.value)[i]=value.real();}
  C arg(Node&n,int p,int point,int j=0){Parent&a=n.parents[p];return get(a.node,a.component,(int)a.map[point*a.width+j]);}
  void put(Node&n,int point,C value){if(n.iscomplex)((C*)n.value)[point]=value;else((double*)n.value)[point]=value.real();}
  void touch(int node,int point){
    if(!inUndo || undoMarks[node][point]==undoEpoch)return;
    undoMarks[node][point]=undoEpoch;journal.push_back({node,point,valid[node][point],oldValues.size()});++savedPoints;
    for(int c=0;c<nodes[node].components;++c)oldValues.push_back(get(node,c,point));
  }
  void begin(){
    if(inUndo)throw std::runtime_error("nested grid transaction");inUndo=true;
    if(++undoEpoch==0){for(auto&m:undoMarks)std::fill(m.begin(),m.end(),0);++undoEpoch;}
    journal.clear();oldValues.clear();oldActive=active;oldPending.clear();
    for(int i:oldActive)oldPending.push_back(pending[i]);
  }
  void rollback(){
    for(auto&s:journal){for(int c=0;c<nodes[s.node].components;++c)set(s.node,c,s.point,oldValues[s.start+c]);valid[s.node][s.point]=s.valid;}
    for(int i:active){pending[i].clear();queued[i]=0;}active=oldActive;
    for(size_t j=0;j<oldActive.size();++j){int i=oldActive[j];pending[i]=oldPending[j];queued[i]=1;}
    journal.clear();oldValues.clear();inUndo=false;
  }
  void queue(int node,int point){
    if(!queued[node]){queued[node]=1;active.push_back(node);}pending[node].push_back(point);
  }
  void clear(){
    // 统计运行阶段实际执行的有效位分支检查；编译可达性和纯赋值不计入。
    for(int i=0;i<count;++i){pending[i].clear();queued[i]=0;
      for(int j=0;j<nodes[i].size;++j){++validityChecks;if(valid[i][j])touch(i,j);valid[i][j]=!live[i][j];}}
    active.clear();
    for(int i=0;i<count;++i)if(nodes[i].opcode!=-1)for(int j=0;j<nodes[i].size;++j)if(live[i][j])queue(i,j);
  }
  void invalidate(int changed_count,const int64_t*changed){
      // 只沿直接标量边传播；无需为每个未知量存储一份传递闭包。
      std::vector<Edge> queue;
      for(int i=0;i<changed_count;++i)for(int64_t j=offsets[changed[i]];j<offsets[changed[i]+1];++j)queue.push_back(edges[j]);
      for(size_t q=0;q<queue.size();++q){Edge e=queue[q];int64_t id=starts[e.node]+e.point;
        for(int64_t j=childOffsets[id];j<childOffsets[id+1];++j){Edge v=children[j];++validityChecks;if(valid[v.node][v.point]){
          touch(v.node,v.point);valid[v.node][v.point]=0;++invalidated;this->queue(v.node,v.point);queue.push_back(v);}}}
  }
  int calculate(int i,std::vector<int>&dirty,Callback callback){
      Node&n=nodes[i];if(dirty.empty())return 1;
      std::sort(dirty.begin(),dirty.end());
      for(int j:dirty)touch(i,j);
      points+=(uint64_t)dirty.size()*n.components;++batches;
      if(n.opcode==0){if(!callback(i,(int)dirty.size(),dirty.data())){if(!inUndo)clear();return 0;}}
      else for(int j:dirty){C a=arg(n,0,j),b=n.parent_count>1?arg(n,1,j):C(0,0),v;
        bool real=!n.iscomplex;
        switch(n.opcode){
        case 1:v=a+b;break;case 2:v=a-b;break;case 3:v=a*b;break;
        case 4:v=real?C(a.real()/b.real(),0):a/b;break;
        case 5:v=real?C(std::pow(a.real(),b.real()),0):std::pow(a,b);break;
        case 6:v=-a;break;case 7:v=std::abs(a);break;case 8:v=a.real();break;case 9:v=a.imag();break;
        case 10:v=(a.real()<b.real());break;case 11:v=(a.real()<=b.real());break;
        case 12:v=(a.real()>b.real());break;case 13:v=(a.real()>=b.real());break;case 14:v=(a==b);break;
        case 15:v=a.real()!=0?b:arg(n,2,j);break;
        case 16:v=std::isnan(a.real())?a:(std::isnan(b.real())?b:(a.real()>b.real()?a:b));break;
        case 17:v=std::isnan(a.real())?a:(std::isnan(b.real())?b:(a.real()<b.real()?a:b));break;
        case 18:v=real?C(std::sqrt(a.real()),0):std::sqrt(a);break;
        case 19:v=real?C(std::log(a.real()),0):std::log(a);break;
        case 20:v=real?C(std::tanh(a.real()),0):std::tanh(a);break;
        case 21:v=a;break;
        case 22:v=0.;for(int k=0;k<n.parents[0].width;++k){double w=n.weights[j*n.parents[0].width+k];if(w!=0)v+=w*arg(n,0,j,k);}break;
        case 23:v=arg(n,(int)n.selector[j],j);break;
        default:if(!inUndo)clear();return -1;
        }put(n,j,v);
      }
      for(int j:dirty)valid[i][j]=1;
      return 1;
  }
  int run(int changed_count,const int64_t*changed,int target_count,const Target*targets,Callback callback){
    // 按原生显式请求的目标条目累计，包含重复请求和失败试探；journal 回退不抹去已发生的工作。
    requested+=(uint64_t)target_count;
    invalidate(changed_count,changed);
    std::sort(active.begin(),active.end());
    for(int i:active)if(calculate(i,pending[i],callback)!=1)return 0;
    for(int i:active){pending[i].clear();queued[i]=0;}active.clear();
    return 1;
  }
};
extern "C" {
// 加载器先核对版本和字段布局，再绑定任何带指针的调度入口。
GRID_EXPORT int GRID_CALL grid_abi_version(){return 2;}
GRID_EXPORT int GRID_CALL grid_abi_layout(int field){
  const size_t layout[]={sizeof(Parent),sizeof(Node),sizeof(Edge),sizeof(Target),sizeof(void*),sizeof(int),sizeof(int64_t),sizeof(double),sizeof(C),offsetof(Parent,map),offsetof(Node,value),offsetof(Node,parents),offsetof(Node,weights),offsetof(Node,selector)};
  return field>=0 && field<(int)(sizeof(layout)/sizeof(layout[0])) ? (int)layout[field] : -1;
}
GRID_EXPORT void* GRID_CALL grid_create_direct(Node*n,int count,const int64_t*off,const Edge*e){auto p=new Program(n,count,off,e);p->buildDirect();return p;}
GRID_EXPORT int GRID_CALL grid_run(void*p,int count,const int64_t*changed,Callback cb){return ((Program*)p)->run(count,changed,0,nullptr,cb);}
GRID_EXPORT int GRID_CALL grid_request(void*p,int count,const int64_t*changed,int nt,const Target*targets,double*out,Callback cb){
  auto*g=(Program*)p;int result=g->run(count,changed,nt,targets,cb);
  if(result==1 && out)for(int i=0;i<nt;++i)out[i]=g->get(targets[i].node,targets[i].component,targets[i].point).real();return result;
}
GRID_EXPORT void GRID_CALL grid_compile_targets(void*p,int n,const Target*t){((Program*)p)->compileTargets(n,t);}
GRID_EXPORT void GRID_CALL grid_begin(void*p){((Program*)p)->begin();}
GRID_EXPORT void GRID_CALL grid_rollback(void*p){((Program*)p)->rollback();}
GRID_EXPORT void GRID_CALL grid_clear(void*p){((Program*)p)->clear();}
GRID_EXPORT void GRID_CALL grid_delete(void*p){delete (Program*)p;}
GRID_EXPORT void GRID_CALL grid_stats(void*p,uint64_t*out){out[0]=((Program*)p)->points;out[1]=((Program*)p)->batches;}
GRID_EXPORT void GRID_CALL grid_stats_detail(void*p,uint64_t*out){auto*g=(Program*)p;out[0]=g->validityChecks;out[1]=g->invalidated;out[2]=g->requested;out[3]=g->savedPoints;out[4]=g->children.size();}
}
