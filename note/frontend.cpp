
/* 2. predictor.scala
    - class BranchPrediction extends BoomBundle: 对每一条指令的预测包含的内容
        - val taken, is_br, is_jal, predicted_pc

    - class BranchPredictionBundle extends BoomBundle with HasBoomFrontendParameters
        - 对整个fetch-width中的指令，进行的转移预测集合，束
        - pc, preds, meta, lhist

    - class BranchPredictionUpdate extends BoomBundle with HasBoomFrontendParameters
        - 对一个取值指令块的分支更新
        - val is_mispredict_update： 用于指示当前是由于转移预测错误的更新
            - 局部预测器推测更新
            - 全局预测器则需要非推测更新
        - val is_repair_update
        - val btb_mispredicts：取指块中是否出现了BTB失效
        - is_btb_mispredict_update = btb_mispredicts =/= 0.U
        - is_commit_update = !(is_mispredict_update || is_repair_update || is_btb_mispredict_update)
        - val pc, br_mask: 取指块中哪一条是分支
        - cfi: control-flow instruction
        - cfi_idx, cfi_taken, cfi_mispredicted, cfi_is_br/jal/jalr
        - ghist: globalhistory, lhist: localhistory
        - val target: cfi jump to
        - val meta = Vec(nBanks, UInt(bpdMaxMetaLength.W))

    class BranchPredictionBankUpdate extends BoomBundle with HasBoomFrontendParameters
        - 对于单独一个bank的转移预测更新
        - 包含的变量和BranchPredictionUpdate一样，但是是以一个bank为单位

    class BranchPredictionRequest extends BoomBundle: 转移预测的请求
        - val pc, ghist
    
    class BranchPredictionBankResponse extends BoomBundle with HasBoomFrontendParameters: 转移预测的应答
        - 没有弄懂为什么是三个，目前推测是if1,if2,if3
        - val f1 = Vec(bankWidth, new BranchPrediction)
        - val f2 = Vec(bankWidth, new BranchPrediction)
        - val f3 = Vec(bankWidth, new BranchPrediction)
        - bankWidth=fetchWidth(1/2)/nbanks, 
        - fetchWidth: Int = if (useCompressed) 2 else 1
          //fetchWidth doubled, but coreInstBytes halved, for RVC:


    class BranchPredictorBank extends BoomModule with HasBoomFrontendParameters
        - 定义模块: 用一个自定义的类来定义模块的
        - 继承自Module类, 有一个抽象字段“io”需要实现, 在类的主构造器里进行内部电路连线
        - val io = IO
            - input: f0_valid, f0_pc, f0_mask
            - input: f1_ghist, f1_lhist (Local history not available until end of f1)
            - input: resp_in (BranchPredictionBankResponse), 
            - input: f3_fire (bool), update(valid(BranchPredictionBankUpdate))
            - output: resp(BranchPredictionBankResponse), 
            - output: f3_meta = Output(UInt(bpdMaxMetaLength.W))
        - val metaSz: metadata size
        - val mems: Seq[Tuple3[String, Int, Int]]
        - def nInputs = 1
        - val s0_idx (f0_pc), s1_idx, s2_idx, s3_idx: RegNext前一个
        - val s0_valid (f0_valid), s1_valid, s2_valid, s3_valid: RegNext前一个
        - val s0_mask (f0_mask), s1_mask, s2_mask, s3_mask
        - val s0_pc (f0_pc), s1_pc
        - val s0_update (update), s1_update
        - val s0_update_idx (fetchIdx(io.update.bits.pc)), s1_update_idx
        - val s0_update_valid (update.valid), s1_update_valid
        - Valid(update): 创建一个Valid接口，对象包括valid(bool, output), bits(T, output)

    class BranchPredictor extends BoomModule with HasBoomFrontendParameters
        - val io = IO
            - input f0_req Valid(new BranchPredictionRequest)
            - input f3_fire = bool
                - 目前看是当f4阶段中f3_bpd_resp.io.enq.fire()时有效
            - input update = Valid(new BranchPredictionUpdate)
            - ouput resp: Bundle
                f1, f2, f3: BranchPredictionBundle
        - val banked_predictors: 创建bankpred的对象列表/数组
        - val banked_lhist_providers: Seq(nbanks), LocalBranchPredictorBank
        - 逻辑部分：
            - 如果仅有一个bank，则根据f0_req的情况设置bank0的预测器和局部历史
            - 否则，最多提供两个bank，并且分配设置每个bank的pc，历史等信息

            - 给出每个阶段的预测结果
            - 将predictor的端口输入中的update信息传递到子bank的信息中的update信息中

    class NullBranchPredictorBank extends BranchPredictorBank
        - val mems = Nil 
        - Nil represents the end of the list

*/

/*  composer.scala
    class ComposedBranchPredictorBank extends BranchPredictorBank
        - val (components, resp) = getBPDComponents(io.resp_in(0), p)
        - 设置components的属性=io.属性
        - components是BranchPredictorBank的数组/列表
        - components在综合之后会包括LoopBP, BTB, TAGE, BIMBranchPredictorBank, FAMicroBTBBranchPredictorBank

        - 所有的子组件的预测器都是采用resp_in来传递信息，预测器的resp=最后一个组件的resp（在configmixins中设置的）
        - 所有子组件仅更新自己需要更新的那一部分，例如f1/f2/f3
            bankpredictor.io.resp := io.resp_in(0)
            ubtb.io.resp_in(0)  := resp_in(predictor中初始化为0)
            bim.io.resp_in(0)   := ubtb.io.resp
            btb.io.resp_in(0)   := bim.io.resp
            tage.io.resp_in(0)  := btb.io.resp
            loop.io.resp_in(0)  := tage.io.resp
            predictor.resp := loop.io.resp
        - 在前端使用预测结果时，如果预测跳转，则需要判断predicted_tag是有效的才行，即taken && btb hit
        - f1的预测有fabtb提供，f2的预测由bim+btb提供， f3的预测由tage+btb(第二级)提供


*/

/* bim.scala: bimodal，第一个周期的预测器，hbim是第二个周期开始的预测器
    class BIMMeta extends BoomBundle with HasBoomFrontendParameters
        - bims = Vec(bankWidth, UInt(2.W))  
        - {bims : UInt<2>[4]}
    
    case class BoomBIMParams
        - nSets: Int = 2048
        - Case classes are good for modeling immutable不可变 data.
        - 创建时不需要使用new关键字
        - 支持比较 == 和 copy功能

    class BIMBranchPredictorBank extends BranchPredictorBank
        - val nWrBypassEntries = 2
        - def bimWrite(v: UInt, taken: Bool)
            - 根据预测结果加减饱和计数器
            - 返回值应该也是加减之后的饱和计数器的值
        - val s2_meta: Wire(new BIMMeta)
            - wire s2_meta : {bims : UInt<2>[4]}
        - val doing_reset: bool reg true
        - val reset_idx: reg, 0, idx位数（log(nSets)）
            - reset_idx := reset_idx + doing_reset
            - 当doing_rest=1时，处理器会逐步更新每一个表项，直到全部更新完
        - val data = SyncReadMem(nSets, Vec(bankWidth, UInt(2.W)))
            - SyncReadMem: 
                - 一个顺序/同步读、顺序/同步写的存储器。
                - 写入在请求后的上升时钟边沿生效。读在请求后的上升沿返回数据。读后写行为（当同一周期内请求对同一地址进行读和写时）未定义
            - UInt<2>[4][2048]
        - val mems = Seq(("bim", nSets, bankWidth * 2))
            - 不知道什么作用，可能是用于计算存储开销的
        
        - val s2_req_rdata: RegNext(data.read(s0_idx   , s0_valid))
            - 从BIM中获取计数器的值
            - UInt<2>[4]：对应于四个端口
        - val s2_resp: Wire(Vec(bankWidth, Bool()))
            - UInt<1>[4]
            - s2_resp(w) := s2_valid && s2_req_rdata(w)(1) && !doing_reset， 预测结果
            - s2_meta.bims(w)  := s2_req_rdata(w)，记录原始的数据，应该用于更新使用
        - val s1_update_wdata: wire, uint<2>[4]
        - val s1_update_wmask = wire, bool[4]
        - val s1_update_meta = {bims : UInt<2>[4]}
        - val s1_update_index = s1_update_idx
        - val wrbypass_idxs = Reg(Vec(nWrBypassEntries, UInt(log2Ceil(nSets).W)))
            - UInt<11>[2], clock
        - val wrbypass: Reg(Vec(nWrBypassEntries, Vec(bankWidth, UInt(2.W))))
            - UInt<2>[4][2], clock
        - val wrbypass_enq_idx: RegInit(0.U(log2Ceil(nWrBypassEntries).W))
            - UInt<1>, clock
        - val wrbypass_hits: UInt<1>[2]
            - 判断s1_update_index和wrbypass_idxs每个元素是否相等
        - val wrbypass_hit = wrbypass_hits.reduce(_||_)
            - reduce(_||_): 归约，相当于wrbypass_hits[0] || wrbypass_hits[1]
        - val wrbypass_hit_idx = PriorityEncoder(wrbypass_hits)
            - mux(wrbypass_hits[0], UInt<1>("h00"), UInt<1>("h01"))
            - // 优先译码器，优先第一个[0]
        - 逻辑部分：
            - 根据s1_update中的信息，更新bim的元数据，包括判断是否是分支，是否跳转
                - s1_update_wmask记录是否要写
                - s1_update_wdata记录要写入什么东西
            - 判断当前是否需要写入BTB，如果需要则写入
                - doing_rest需要更新BTB
                - commit阶段需要更新BTB，根据s1_update_index， s1_update_wdata，s1_update_wmask
                - 单独需要判断bypass的情况，将bypass的信息继续下来
*/

/* btb.scala
    case class BoomBTBParams
        - nSets 128
        - nWays: 2
        - offsetSz: 13
        - extendedNSets: 128

    class BTBBranchPredictorBank(params: BoomBTBParams) extends BranchPredictorBank
        - val tagSz, offsetSz, extendedNSets
        - class BTBEntry extends Bundle
            - offset  : SInt(offsetSz.W)
            - extended : bool
        - class BTBMeta extends Bundle 
            - val is_br = Bool()
            - val tag   = UInt(tagSz.W)
        - class BTBPredictMeta extends Bundle
            - val write_way = UInt(log2Ceil(nWays).W)   //1bit, 判断当前写入哪一路？
        - val s1_meta : wire(new BTBPredictMeta)
            - {write_way : UInt<1>}
            - 用于记录哪一路需要更新
        - val f3_meta = RegNext(RegNext(s1_meta))
            - io.f3_meta := f3_meta.asUInt
        - val doing_reset, reset_idx
        
        - val meta: Seq.fill(nWays) { SyncReadMem(nSets, Vec(bankWidth, UInt(btbMetaSz.W))) }
            - meta_0 : UInt<31>[4][128]
            - meta_1 : UInt<31>[4][128]
            - nSets=128, bankwidth=4, btbMetaSz=31, tag=30
        - val btb: Seq.fill(nWays) { SyncReadMem(nSets, Vec(bankWidth, UInt(btbEntrySz.W))) }
            - btb_0/1 : UInt<14>[4][128]
            - offsetsz = 13
        - val ebtb: SyncReadMem(extendedNSets, UInt(vaddrBitsExtended.W))
            - UInt<40>[128]
            - 偏移+PC不够标识了，则需要使用扩展位？
        
        - val s1_req_rbtb:
            - {offset : SInt<13>, extended : UInt<1>}[4][2]
            - 从128组中获取对应的一组的两路，四个bank？
        - val s1_req_rmeta:
            - s1_req_rmeta : {is_br : UInt<1>, tag : UInt<30>}[4][2]
        - val s1_req_rebtb: ebtb[s0_idx.bits(6,0)]
            - when io.f0_valid : @[btb.scala 75:31]
                _WIRE_40 <= s0_idx @[btb.scala 75:31]
                node _T_51 = or(_WIRE_40, UInt<7>("h00")) @[btb.scala 75:31]
                node _T_52 = bits(_T_51, 6, 0) @[btb.scala 75:31]
                read mport s1_req_rebtb = ebtb[_T_52], clock
        - val s1_req_tag = s1_idx >> log2Ceil(nSets)
        - val s1_resp   = Wire(Vec(bankWidth, Valid(UInt(vaddrBitsExtended.W))))
            - {valid : UInt<1>, bits : UInt<40>}[4]
        - val s1_is_br, s1_is_jar: Wire(Vec(bankWidth, Bool()))
        - val s1_hit_ohs: UInt<1>[2][4]
            - 对比s1_req_rmeta是否和s1_req_tag(tagSz-1,0)一样
        - val s1_hits: s1_hit_ohs.map { oh => oh.reduce(_||_) }
            - 判断每个bank是否命中
            - s1_hits_0 = or(s1_hit_ohs[0][0], s1_hit_ohs[0][1])
            - s1_hits_1， s1_hits_2，s1_hits_3
        - val s1_hit_ways: s1_hit_ohs.map { oh => PriorityEncoder(oh) }
            - 判断具体命中的是哪一路（四个bank）

        - 逻辑部分：第一个循环
            - 遍历获取命中BTB的信息，包括计算target地址，填充到s1_resp
            - 传递信息到f2和f3的流水级中

        - 插入新的BTB表项
        - val alloc_way: 确定分配哪一路，通过一种奇怪的方法，确定多路时替换哪一路
        - s1_meta.write_way： 判断是更新命中的那一路，还是未命中替换的那一路
        
        - val s1_update_cfi_idx = s1_update.bits.cfi_idx.bits
        - s1_update_meta    = s1_update.bits.meta.asTypeOf(new BTBPredictMeta)
            - {write_way : UInt<1>}
        - max_offset_value，min_offset_value：最大和最小偏移量
        - new_offset_value：pc和target之间的偏移量
        - offset_is_extended：判断是否需要使用扩展的地址位
        - val s1_update_wbtb_data
        - val s1_update_wbtb_mask: 更新btb的掩码，即是否需要写入btb
        - val s1_update_wmeta_mask: 更新btb元数据的掩码，即是否需要写入btb
        - val s1_update_wmeta_data

        - 更新BTB和BTBmeta，主要是确定更新哪些地方，更新什么数据
        - 疑问：四个bank存储的数据是一样的，tag基本一样（预测错了就删掉）

        - 整个的预测过程：
            - 每次提供四条指令的预测结果，一个取指块，分四个部分，对应四个bank
            - 预测时需要检查是否为br，tag对应否，跳转地址（扩展还是pc+off）
*/  


/* ras.scala
    class BoomRAS extends BoomModule
        - io： RAS的top信息由外界维护
            - input: read_idx, 指明当前top的位置
            - ouput: read_addr, 指令当前top存储的内容
            - input: write_valid， 是否写入
            - input: write_idx: 写入的位置
            - input: 写入的内容
        - val ras: Reg(Vec(nRasEntries, UInt(vaddrBitsExtended.W)))
        - 逻辑：读取地址
            - 需要检测上一个周期是否发生了写入，如果是，则直接将要写入的地址旁路给结果（应该是周期上的问题）
        - 逻辑：写入地址

*/

/* faubtb.scala FA Micro BTB， 全相联BTB
    case class BoomFAMicroBTBParams: ubtb
        - nWays=16, offsetSz=13
        - 路数也太多了
    
    //只有一组16路，可能是对应mirco-op？
    class FAMicroBTBBranchPredictorBank extends BranchPredictorBank
        - val tagSz, nWrBypassEntries=2，支持的bypass的数量
        - def bimWrite(v: UInt, taken: Bool): UInt
            - 两位饱和计数器的更新
        
        - class MicroBTBEntry extends Bundle
            - offset = SInt(offsetSz.W)
        - class MicroBTBMeta extends Bundle
            - is_br : bool
            - tag: UInt(tagSz.W)
            - ctr: 2bit，饱和计数器，用于判断是否跳转？
        - class MicroBTBPredictMeta extends Bundle
            - hits: Vec(bankWidth, Bool()), 是否发生了命中
            - write_way: 命中了哪一路

        - val s1_meta: MicroBTBPredictMeta
        - val meta: MicroBTBMeta[4][16]
            - reg, {is_br : UInt<1>, tag : UInt<37>, ctr : UInt<2>}[4][16], clock
        - val btb: MicroBTBEntry[4][16]
            - {offset : SInt<13>}[4][16], clock
        
        - val s1_req_tag   = s1_idx, 访问的索引，应该是pc
        - val s1_resp = {valid : UInt<1>, bits : UInt<40>}[4]
            - Wire(Vec(bankWidth, Valid(UInt(vaddrBitsExtended.W))))
            - vaddrBitsExtended = 40
        - val s1_taken, s1_is_br, s1_jal: bool[4]
       
        - val s1_hit_ohs：判断在哪一路命中了，tag和meta进行对比
            - UInt<1>[16][4]
        - val s1_hits: 是否命中，s1_hits[4], bankwidth
        - val s1_hit_ways: 在那一路命中了，s1_hit_ways[4], bankwidth
            - 逻辑部分，利用s1_hit_ways信息，更新s1_resp，s1_taken，s1_is_br等的信息
            - 更新io_resp.f1，f2,,f3的信息，f2延后一个周期，f3延后f2一个周期
            - 更新io.f3_meta = RegNext(RegNext(s1_meta.asUInt))，应该是为了在f3周期更新使用 

        - val alloc_way: 利用异或的方式计算得到一个插入的表项位置信息
            - 逻辑部分: s1_meta.write_way
                - 如果命中，则选择命中的路，否则使用分配得到的路，表明要写
        
        - 逻辑部分：
            - val s1_update_cfi_idx :只有两位，应该是bankwidth的索引
            - val s1_update_meta, s1_update_write_way
            - val max_offset_value, min_offset_value, new_offset_value，计算偏移，前两者没有用
            - val s1_update_wbtb_data, s1_update_wbtb_mask, s1_update_wmeta_mask
            - 更新BTB和btbmeta
            - 基本类似于btb，但是结构有些许差别
*/

/* ubtb.scala micro btb

    case class BoomMicroBTBParams
        - nSets: Int = 256,
        - offsetSz: Int = 13

    //只有nset，没有路，即直接映射的BTB
    - class MicroBTBBranchPredictorBank extends BranchPredictorBank
        - val tagSz, offsetSz, nWrBypassEntries
            - tagSz =  vaddrBitsExtended - log2Ceil(nSets) - log2Ceil(fetchWidth) - 1
                = 40 - 8 - 2 -1 = 29
        - def bimWrite: 更新饱和计数器
        - doing_reset, reset_idx
        
        - class MicroBTBEntry extends Bundle
            - val offset
        - class MicroBTBMeta extends Bundle
            - val is_br: bool
            - val tag
            - val ctr uint<2> 饱和计数器
        - class MicroBTBPredictMeta extends Bundle
            - ctrs: Vec(bankWidth, UInt(2.W))
            - uint<2>[4]
        - val s1_meta = MicroBTBPredictMeta

        - val meta = uint<MicroBTBMeta>[4][256]
        - val btb = uint<MicroBTBEntry>[4][256]

        //请求访问btb时的临时寄存器，用于存储部分信息
        - val s1_req_rbtb, s1_req_rmeta, s1_req_tag
            - s1_req_tag=s1_idx >> log2Ceil(nSets)
            - s0_idx       = fetchIdx(io.f0_pc)
            - s1_idx       = RegNext(s0_idx)

        //发送给s1的response
        - val s1_resp: uint<40>[4], 目标地址
            - Vec(bankWidth, Valid(UInt(vaddrBitsExtended.W))
        - val s1_taken, s1_is_br, s1_is_jal: bool
            - 逻辑部分：
                - 填充s1_resp，判断是否命中valid，pc+offset
                - 更新s1_taken, s1_is_br, s1_is_jal
                - 记录元数据到s1_meta.ctrs
            - 逻辑部分：更新output接口部分
                - 更新f1，设定f2和f3为RegNext
        
        //更新btb和元数据的内容
        - val s1_update_cfi_idx, s1_update_meta(MicroBTBPredictMeta)
        - val wrbypass_idxs, wrbypass(MicroBTBMeta[bankWidth][nWrBypassEntries]), wrbypass_enq_idx
        
        - val wrbypass_hits, wrbypass_hit, wrbypass_hit_idx
            - 判断更新的数据是否和bypass的重合
            - io.update > s0_update(T=1) > s1_update(T=2) 

        - val new_offset_value: 计算需要更新的offset
        - val s1_update_wbtb_data     = Wire(new MicroBTBEntry)
            - s1_update_wbtb_data.offset := new_offset_value
        - val s1_update_wbtb_mask： 写入哪一个bank的元素

        - val s1_update_wmeta_data, s1_update_wmeta_mask
            - 临时寄存器，用于存储需要更新的元数据的内容

        - 逻辑部分：
            - 更新btb和meta
            - 更新旁路表中的内容


*/


/* hbim.scala history bim, 用于第二级，目前推测是用于构建局部/全局/gshare预测器
    case class BoomHBIMParams
        - nSets: Int = 2048,
        - useLocal: Boolean = false,
        - histLength: Int = 32

    //基本类似于bim，但是索引方式有一些不同
    class HBIMBranchPredictorBank extends BranchPredictorBank
        - val nWrBypassEntries = 2
        - def bimWrite: 饱和计数器更新
        - val doing_reset, reset_idx：预测表的重置

        - val s3_meta: BIMMeta, 用于第三个周期更新
        - val data UInt<2>[2048][4]

        - val f1_idx: 计算预测表的索引，局部/全局历史
        - val s3_req_rdata: 预测表的访问结果，uint<2>[4]
        - val s3_resp: bool[4]
            - 逻辑：s3_resp更新为预测表项计数器的最高位
            - 逻辑：并且在s3_meta中记录元数据，用于之后的更新

        //计算更新时的数据，元数据，掩码和索引
        - val s1_update_wdata： uint<2>[4]
        - val s1_update_wmask: bool[4], 是否需要更新
        - val s1_update_meta: 元数据
        - val s1_update_index: 更新时的索引

        //bypass的信息，包括之后用于更新bypass和从bypass中获取信息
        - val wrbypass_idxs: 每个bypass表项对应于预测表的哪一个表项idx
        - val wrbypass: bypass的内容，计数器值
        - val wrbypass_enq_idx: 进入bypass的索引
        - val wrbypass_hits: bool[nWrBypassEntries], 判断需要更新的表项是否正在bypass中
        - val wrbypass_hit: wrbypass_hits.reduce(_||_)
        - val wrbypass_hit_idx: bypass中命中了哪个位置

        - 逻辑部分：更新bim和元数据到临时寄存器，s1_update_wmask，s1_update_wdata
            - 判断需要更新的指令是否发生了跳转
            - 获取对应表项的旧值，如果bypass中有，则从bpyass中获取
        
        - 逻辑部分：实际修改data的值
            - 根据s1_update_wmask和s1_update_wdata修改data

        - 逻辑部分：更新bypass的内容
            - 如果bypass命中了，则记录当命中的位置
            - 否则新增加表项记录更新的数据

        - 逻辑部分： 将预测信息和元数据信息传递到f3中
            - io.resp.f3(w).taken := s3_resp(w)
            - io.f3_meta := s3_meta.asUInt
*/

/* local.scala
    abstract class AbstractLocalBranchPredictorBank extends BoomModule  with HasBoomFrontendParameters
        - io
            -input: f0_valid, bool
            -input: f0_pc, uint

            -output: f1_lhist, UInt(localHistoryLength.W)
            -output: f3_lhist, UInt(localHistoryLength.W)

            -input: f3_taken_br, bool
            -input: f3_fire, bool
            -update: input, bundle
                - valid, mispredict, repair: bool
                - pc, lhist = uint

    class NullLocalBranchPredictorBank extends AbstractLocalBranchPredictorBank
        - io.f1_lhist := 0.U
        - io.f3_lhist := 0.U

    //这个类应该只负责管理局部预测器的局部历史表，输出仅为根据pc找到的局部历史
    class LocalBranchPredictorBank extends AbstractLocalBranchPredictorBank
        - val nSets: localHistoryNSets
        - val doing_reset, reset_idx
        
        - val entries: uint<localHistoryLength>[nSets], 局部历史表

        //def fetchIdx(addr: UInt) = addr >> log2Ceil(fetchBytes)
        - val s0_idx = fetchIdx(io.f0_pc)
        - val s1_rhist = entries.read(s0_idx, io.f0_valid)
        - val s2_rhist = RegNext(s1_rhist)
        - val s3_rhist = RegNext(s2_rhist) //随着流水级向后传递
            - io.f1_lhist := s1_rhist
            - io.f3_lhist := s3_rhist
        
        //update
        - val f3_do_update: bool， false
        - val f3_update_idx: 更新时访问的索引
        - val f3_update_lhist: 更新需要写入的数据
        - val s1_update: RegNext(io.update)
        - val s1_update_idx: fetchIdx(s1_update.pc)

        - 逻辑部分1： 
            - 当s1_update有效，并且s1_update表明预测错误或者需要修复时
                - 将f3_do_update设置为true
                - 更新f3_update_idx和f3_update_lhist的内容设置为s1_update中的内容
            - 否则，当io.f3_fire为1时，更新，应该指代的时第三个周期发现没啥问题
                - - 将f3_do_update设置为true
                - f3_update_idx   := RegNext(RegNext(RegNext(s0_idx)))
                - f3_update_lhist := s3_rhist << 1 | io.f3_taken_br
        - 逻辑部分2：
            - 重置时的更新
            - f3_do_update为true的更新

*/


/* loop.scala
    case class BoomLoopPredictorParams
        - nWays: 4
        - threshold: 7

    class LoopBranchPredictorBank extends BranchPredictorBank
        - val tagSz=10, nSets=16
        - class LoopMeta extends Bundle
            - val s_cnt   = UInt(10.W)
        - class LoopEntry extends Bundle
            - val tag   = UInt(tagSz.W)
            - val conf  = UInt(3.W)     
            - val age   = UInt(3.W)
            - val p_cnt = UInt(10.W)    //应该一个是目标循环次数，一个是当前次数
            - val s_cnt = UInt(10.W)
        
        - class LoopBranchPredictorColumn extends Module
            - io
                - input: f2_req_valid, f2_req_idx, f2_req_idx
                - input: f3_pred_in
                - ouput: f3_pred, f3_meta(LoopMeta)
                - input: update_mispredict, update_repair: bool
                - input: update_idx
                - input: update_resolve_dir,
                - input: update_meta(LoopMeta)
            - val doing_reset, reset_idx
            
            - val entries: Reg[LoopEntry][16]
            //第二个周期获取entry，第三个周期给出预测结果
            - val f2_entry: entries(io.f2_req_idx)
                - 根据io.update_idx等情况，判断是否需要修改f2_entry
                - 有点类似于bypass
            - val f3_entry = RegNext(f2_entry)
            - val f3_scnt  
                - 判断这个周期需要修改的位置是否和上一个周期的f2_entry是一个，如果时，需要修改f3_scnt为修改之后的值
            - val f3_tag = RegNext(io.f2_req_idx(tagSz+log2Ceil(nSets)-1,log2Ceil(nSets)))
                - RegNext(bits(io.f2_req_idx, 13, 4))
                - 根据f2的idx，记录tag
            - 逻辑部分1：
                - 更新io的f3_pred=f3_pred_in
                - 更新io的f3_meta.s_cnt := f3_scnt
            - 逻辑2：在f3给出预测结果
                - 根据req_idx的tag和f2_entry的tag进行对比，判断是否命中
                - 如果命中，并且f3_scnt=pcnt, 可信度足够，则预测发生跳转

            //第三个周期预测，第四个周期更新？
            - val f4_fire  = RegNext(io.f3_req_fire)
            - val f4_entry = RegNext(f3_entry)
            - val f4_tag   = RegNext(f3_tag)
            - val f4_scnt  = RegNext(f3_scnt)
            - val f4_idx   = RegNext(RegNext(io.f2_req_idx))
            - 逻辑：
                - 如果entry是命中中，并且已经达到循环退出条件，则更新entries对应的表项，scnt=0，age=7
                - 否则更新entries对应的表项，scnt+1， age+1

            //预测错误时的更新
            - val entry     =entries(io.update_idx)
            - val tag       =io.update_idx(13, 4) bits
            - val tag_match =entry.tag === tag
            - val ctr_match =entry.p_cnt === io.update_meta.s_cnt  ?
            - val wentry = WireInit(entry)
            - 更新逻辑：预测错误，并且当前尚未处于重置状态
                - 可信度=7，tag匹配，则scnt=0, conf=0
                - entry.conf === 7.U && !tag_match，尚未处理
                - entry.conf =/= 0.U(!=) && tag_match && ctr_match
                    - conf+1,scnt=0
                - entry.conf =/= 0.U && tag_match && !ctr_match
                    - conf=0, scnt=0, pcnt=update_meta.scnt
                - entry.conf =/= 0.U && !tag_match && entry.age === 0.U
                    - 插入新的表项
                    - tag=tag, conf=1, scnt=0, pcnt=meta.scnt
                - entry.conf =/= 0.U && !tag_match && entry.age =/= 0.U
                    - 不插入表项，但是age-1
                - entry.conf === 0.U && tag_match && ctr_match
                    - conf=1, age=7, scnt=0, 即保证不会立马被替换出去
                - entry.conf === 0.U && !tag_match
                    - 插入新的表项
                    - tag=tag, conf=1, scnt=0, age=7, pcnt=meta.scnt
            - 更新逻辑：update_repair
                - 当tag匹配并且!(f4_fire && io.update_idx === f4_idx)
                    scnt=meta.scnt

        - 创建loopBP
            - columns =   LoopBranchPredictorColumn[bankWidth]
            - f3_meta = LoopMeta[bankWidth]
            - update_meta : LoopMeta[bankWidth]
            - 逻辑部分：
                - 设置columns的一些IO端口信息
                - 更新f3_meta为每个LBP的f3_meta


*/


/* tourney.scala alpha21264锦标赛预测器的选择器实现
    case class BoomTourneyBPDParams
        - nSets: Int = 128,
        - histLength: Int = 32

    class TourneyBranchPredictorBank extends BranchPredictorBank
        - def bimWrite(v: UInt, taken: Bool): UInt，饱和计数器更新
        - def compute_folded_hist(hist: UInt, l: Int)
            - 折叠历史到固定的位数
        
        - val doing_reset, reset_idx

        - class TourneyMeta extend Bundle
            - ctrs: uint<2>[bankwidth]
        
        - val s3_meta(TourneyMeta)
        - val data: uint<2>[128][4]

        - val f1_req_idx = f1_ghist ^ s1_idx(pc)
        - val s3_req_rdata = RegNext(data.read(f1_req_idx))
            - 需要等待一个周期才能够读取到数据
        - val s3_resp: bool[4]

        //设置输出端口信息
        - 逻辑部分1：
            - io.resp := io.resp_in(0)
            - s3_resp(w) := Mux(s3_req_rdata(w)(1), io.resp_in(1).f3(w).taken, io.resp_in(0).f3(w).taken)  
                - 根据计数器最高位判断当前选择哪一个子预测器的预测结果
            - s3_meta.ctrs(w) := s3_req_rdata(w)，记录元数据信息，用于之后的更新
                - io.f3_meta := s3_meta.asUInt， 更新输出端口
            - io.resp.f3(w).taken := s3_resp(w)， 更新输出端口的响应接口

        //更新选择器的状态
        - val s1_update_wdata: uint<2>[4]
        - val s1_update_wmask: bool[4]
        - val s1_update_meta (TourneyMeta)
        - val s1_update_index: s1_update.bits.ghist ^ s1_update_idx
        - 逻辑部分1：
            - 根据预测的正确与否，以及当前选择的子预测器来更新选择器的计数器
            - 根据当前计数器最高位取反，即因为根据最高位确定子预测器，因此错了就要取反，向另一个方向更新
        - 逻辑部分2：
            - 实际写入data中，包括reset和更新

*/

/* tage.scala
    class TageResp extends Bundle
        - val ctr: uint<3>
        - val u: uint<2>
    
    class TageTable(val nRows, val tagSz, val histLength, val uBitPeriod) extends BoomModule()(p)
        - val nWrBypassEntries = 2
        - io:
            -input: f1_req_valid, f1_req_pc, f1_req_ghist
            -ouput: f3_resp = Output(Vec(bankWidth, Valid(new TageResp)))

            -input: update_mask, update_taken, update_alloc(bool), update_old_ctr
            -input: update_pc, update_hist
            -input: update_u_mask(bool[bankw]), update_u(uint<2>[bankw])
        
        - def compute_folded_hist(hist: UInt, l: Int): 折叠历史到固定位数
        - def compute_tag_and_hash(unhashed_idx: UInt, hist: UInt)
            - 根据历史和哈希值来计算idx和tag
        - def inc_ctr(ctr: UInt, taken: Bool): 饱和计数器更新

        - val doing_reset, reset_idx

        - class TageEntry extends Bundle
            - val valid = Bool() 
            - val tag = UInt(tagSz.W)
            - val ctr = UInt(3.W)
        
        //实际存储
        - val hi_us: bool[4][nRows]
        - val lo_us: bool[4][nRows]
        - val table: uint<tageEntrySz>[4][nRows]

        - val s1_hashed_idx, s1_tag
            - 根据f1_req_pc和f1_req_ghist计算索引和tag
            - 第一个阶段的任务
        //第二个阶段读取实际的数据
        - val s2_tag = RegNext(s1_tag)
        - val s2_req_rtage: table.read
        - val s2_req_rhius: hi_us.read
        - val s2_req_rlous: lo_us.read
        - val s2_req_rhits: valid && tagmatch
        - 更新输出端口的f3_resp，传递到第三个周期
            //valid即表示了是否命中
            - io.f3_resp(w).valid    := RegNext(s2_req_rhits(w))
            - io.f3_resp(w).bits.u   := RegNext(Cat(s2_req_rhius(w), s2_req_rlous(w)))
            - io.f3_resp(w).bits.ctr := RegNext(s2_req_rtage(w).ctr)


        //更新表的内容，包括used表的内容
        //获取当前应该刷新used的表项和刷新高低位信息
        - val clear_u_ctr：RegInit(0.U((log2Ceil(uBitPeriod) + log2Ceil(nRows) + 1).W))
            - 用于周期性的刷新表中某个表项的used位域的内容呢
            - 每周期+1
        - val doing_clear_u: clear_u_ctr(log2Ceil(uBitPeriod)-1,0) === 0.U
            - 根据低log2Ceil(uBitPeriod)来判断是否到了更新周期
            - 此时在根据高位作为索引进行刷新
        - val doing_clear_u_hi
        - val doing_clear_u_lo
            - 根据clear_u_ctr的最高位来决定是刷新used的高位还是低位
        - val clear_u_idx = clear_u_ctr >> log2Ceil(uBitPeriod)
            - 刷新的idx

        - val update_idx, update_tag
            - 根据io.update_pc, io.update_hist计算tag和idx
        - val update_wdata (TageEntry)
            - 临时寄存器，存储需要更新的具体数据是什么
        - 逻辑部分：分别更新table, hi_us, lo_us

        // bypass更新
        - val wrbypass_tags    = Reg(Vec(nWrBypassEntries, UInt(tagSz.W)))
        - val wrbypass_idxs = Reg(Vec(nWrBypassEntries, UInt(log2Ceil(nRows).W)))
        - val wrbypass = Reg(Vec(nWrBypassEntries, Vec(bankWidth, UInt(3.W))))
        - val wrbypass_enq_idx = RegInit(0.U(log2Ceil(nWrBypassEntries).W))
        - val wrbypass_hits: 比较update数据的tag和idx
        - val wrbypass_hit = wrbypass_hits.reduce(_||_)
        - val wrbypass_hit_idx
        - 逻辑部分1：
            - 根据bypass信息和跳转信息，是否分配表项信息来更新update_wdata(w).ctr
            - 更新update_wdata(w)的valid, tag
            - update_hi_wdata(w)    := io.update_u(w)(1)
            - update_lo_wdata(w)    := io.update_u(w)(0)
        - 逻辑部分2： 更新bypass的内容


    case class BoomTageParams
        //                                           nSets, histLen, tagSz
        - tableInfo: Seq[Tuple3[Int, Int, Int]] = Seq((  128,       2,     7),
                                              (  128,       4,     7),
                                              (  256,       8,     8),
                                              (  256,      16,     8),
                                              (  128,      32,     9),
                                              (  128,      64,     9)),

        - uBitPeriod: Int = 2048

    
    class TageBranchPredictorBank(BoomTageParams) extends BranchPredictorBank
        - val tageUBitPeriod, tageNTables=params.tableInfo.size

        - class TageMeta extends Bundle
            - val provider = Vec(bankWidth, Valid(UInt(log2Ceil(tageNTables).W)))
                - 提供预测表是哪一个，valid则表示是否命中过
            - val alt_differs   = Vec(bankWidth, Output(Bool()))
                - 次一级的结果是否和最终结果相同
            - val provider_u    = Vec(bankWidth, Output(UInt(2.W)))
                - used
            - val provider_ctr  = Vec(bankWidth, Output(UInt(3.W)))
            - val allocate = Vec(bankWidth, Valid(UInt(log2Ceil(tageNTables).W)))
                - valid表示是否有合适的表用于分配
                - bits表示分配到哪一个表
        
        
        - def inc_u(u: UInt, alt_differs: Bool, mispredict: Bool)
            - 更新used位域

        - val tables: 根据tableinfo创建了tage的预测表
        - val f3_meta(TageMeta)
        - val f3_resps = VecInit(tables.map(_.io.f3_resp))
            - 每一个表的回应组合在一起
        
        - val s1_update_meta (TageMeta)
        - val s1_update_mispredict_mask
        - val s1_update_taken   = Wire(Vec(tageNTables, Vec(bankWidth, Bool())))
        - val s1_update_old_ctr=Wire(Vec(tageNTables, Vec(bankWidth, UInt(3.W))))
        - val s1_update_alloc  = Wire(Vec(tageNTables, Vec(bankWidth, Bool())))
        - val s1_update_u   = Wire(Vec(tageNTables, Vec(bankWidth, UInt(2.W))))

*/


/*
trait  HasBoomFrontendParameters extends HasL1ICacheParameters
    - val nBanks, bankBytes(每个bank取指大小), bankWidth=fetchWidth(1/2)/nbanks, 
        - nBanks=1, fetchwidth=4
    - val numChunks = cacheParams.blockBytes / bankBytes
    - bank(addr: UInt): 判断地址访问哪一个bank
    - isLastBankInBlock(addr: UInt): 
    - mayNotBeDualBanked(addr: UInt): 当前是否为最后一个bank，是则要从头开始
    - blockAlign(addr: UInt): 块地址对齐，抹除低6位，64B
    - bankAlign(addr: UInt): bank地址对齐
    - fetchIdx(addr: UInt): 获取需要取的块的地址
    - nextBank(addr: UInt), nextFetch(addr: UInt)
    - fetchMask(addr: UInt): 取指地址的掩码
    - bankMask(addr: UInt): bank的掩码
*/


/* fetch-target-queue.scala
    // Each entry in the FTQ holds the fetch address and branch prediction snapshot state.

    case class FtqParameters
        - nEntries: 16
    
    class FTQBundle extends BoomBundle with HasBoomFrontendParameters
        - val cfi_idx   = Valid(UInt(log2Ceil(fetchWidth).W))
            - 取指块中的cfi指令的位置
        - val br_mask: UInt(fetchWidth.W)
        - val cfi_taken, cfi_mispredicted
        - val cfi_type: UInt(CFI_SZ.W), jar/br/jmp之类的
        - val cfi_is_call, cfi_is_ret: 注释中用了像这个词
        - val cfi_npc_plus4: 判断是应该+4还是+2，压缩不压缩的问题
        - val ras_top: UInt(vaddrBitsExtended.W)
        - val ras_idx
            - 当前看到的ras的顶部是什么
        - val start_bank: uint<1>
            - 从哪个bank开始？
        
    class GetPCFromFtqIO extends BoomBundle
        - 提供一个IO，用于为功能单元获取一条指令的pc，包括npc
        - input: ftq_idx: ftq的索引
        - output: entry(FTQBundle)
        - output: ghist(GlobalHistory)
            - ifu/frontend.scala:45
        - output: pc, com_pc
        - output: next_val(bool), 判断npc是否有效
        - output: next_pc

    //用于存储fetchPC和一些转移预测器相关的信息，主要是在流水线中的分支指令的信息
    //目前猜测是boom中Branch rob的功能实现
    class FetchTargetQueue  extends BoomModule with HasBoomCoreParameters with HasBoomFrontendParameter
        - val num_entries = ftqSz, idx_sz=log2Ceil(num_entries)
        - io:IO
            - Flipped: enq=Decoupled(new FetchBundle())
                - 每个取指周期进入一个表项
            - ouput: enq_idx, 返回插入的ftq表项位置
                - io.enq_idx = enq_ptr.
            - Flipped: deq=Valid(UInt(idx_sz.W))
                - 出队列提供的信息，是否有效及表项索引
            - get_ftq_pc: Vec(2, new GetPCFromFtqIO())
                - 提供两个外界查询的端口，通过idx查询记录的信息
            
            - input: debug_ftq_idx, debug_fetch_pc
                - Used to regenerate PC for trace port stuff in FireSim
            
            - input: redirect, Valid(UInt(idx_sz.W)), 用于重定向
            - input: brupdate(BrUpdateInfo), 
            - output: bpdupdate, Valid(new BranchPredictionUpdate)
            - output: ras_update(bool), ras_update_idx(UInt(log2Ceil(nRasEntries).W))
            - output: ras_update_pc
        
        - val bpd_ptr, deq_ptr, enq_ptr: RegInit(0.U(idx_sz.W))
            - deq_ptr=0, enq_ptr=1
            - 各种指针
            - deq_ptr := io.deq.bits, 当io.deq.valid时
            - bpd_ptr: 应该是用于指示当前流水线中最老的分支在表项中的位置
                -  //当前ftq的head位置，只有发生正常的指令提交时才会增加
        - val full: enq_ptr+2 == bpd_ptr||enq_ptr == bpd_ptr
            - 用于判断ftq是否已经满了？
        
        //数据存储定义
        - val pcs: reg, uint<vaddrBitsExtended.W>[num_entries]
            - UInt<40>[32]
        - val meta: mem, uint<bpdMaxMetaLength.W>[nBanks][num_entries]
            - UInt<120>[1][32]
        - val ram: reg, FTQBundle[num_entries]
        - val ghist: 
            - ghist_0 ghist_1
            - ghist_0: mems, GlobalHistory[num_entries]
        - val lhist: 如果使用局部历史，则分配一个空间
            - mem, uint<localHistoryLength.W>[nBanks][num_entries]

        //进入ftq
        - val do_enq = io.enq.fire()
            - = io.enq.ready && io.enq.valid
            - ready = !full || commit_update
            - valid = frontEnd设置
        - val prev_ghist(GlobalHistory), prev_entry(FTQBundle), prev_pc
        - 逻辑部分1：
            - 根据enq中的pc，更新pcs，pcs(enq_ptr) := io.enq.bits.pc
            - val new_entry(FTQBundle), 用于记录enq中的信息，初始化一个表项
            - val new_ghist, 根据io.enq.bits.ghist.current_saw_branch_not_taken来判断当前应该选择哪一个ghist，一个是enq的ghist，一个是根据prev_entry更新的ghist
                - current_saw_branch_not_taken ?
            - 更新lhist,ghist, meta, ram的内容
            - 更新prev_pc,entry,ghist寄存器的内容
            - enq_ptr更新
                - io.enq_idx = enq_ptr.


        - val first_empty : RegInit(true), 表示FTQ初始为空的情况
        //当知道了CFI的跳转目标后，更新转移预测器
        - val ras_update(bool, false)， ras_update_pc, ras_update_idx
            - 和io端口中对应的output端口相连
        - val bpd_update_mispredict, bpd_update_repair: false
        - val bpd_repair_idx, bpd_end_idx: UInt(log2Ceil(ftqSz).W)
        - val bpd_repair_pc
            - output: bpdupdate, Valid(new BranchPredictionUpdate)
        - val bpd_idx
            - = io.redirect.bits, 如果io.redirect.valid
            - = bpd_repair_idx, 如果需要更新修复/错误预测
            - = bpd_ptr
        - val bpd_entry = RegNext(ram(bpd_idx)) //读取相应的数据
        - val bpd_ghist = ghist(0).read(bpd_idx, true.B)
        - val bpd_lhist = lhist.get.read(bpd_idx, true.B)
        - val bpd_meta  = meta.read(bpd_idx, true.B)
        - val bpd_pc    = RegNext(pcs(bpd_idx))
        - val bpd_target = RegNext(pcs(WrapInc(bpd_idx, num_entries)))
            - bpd_idx+1即为下一次取指的信息

        - 寄存器的更新逻辑
            - io.redirect.valid = true  //需要重定向时
                - bpd_update_mispredict := false.B
                - bpd_update_repair     := false.B
            - RegNext(io.brupdate.b2.mispredict) = true //上个周期出现了错误预测的信息
                - bpd_update_mispredict := true.B
                //获取上周期的信息
                - bpd_repair_idx        := RegNext(io.brupdate.b2.uop.ftq_idx)
                - bpd_end_idx           := RegNext(enq_ptr)
            - bpd_update_mispredict = true  //这个周期需要解决预测错误的问题
                - bpd_update_mispredict := false.B
                - bpd_update_repair     := true.B   //需要修复
                - bpd_repair_idx        := WrapInc(bpd_repair_idx, num_entries)
            - bpd_update_repair && RegNext(bpd_update_mispredict) 
            //看起来和上一个的意思一样，只不过是下一个周期的事
                - bpd_repair_pc         := bpd_pc
                - bpd_repair_idx        := WrapInc(bpd_repair_idx, num_entries)
            - bpd_update_repair
                - bpd_repair_idx        := WrapInc(bpd_repair_idx, num_entries)
                - bpd_update_repair := false.B （如果ftq到头了或者没得需要修复）

        //设置io端口中的bpdupdate信息，用于之后的预测器更新？
        - val do_commit_update: 判断是否处于正常的commit更新状态
            - 要求没有misp，repair，重定向等等问题
        - val do_mispredict_update = bpd_update_mispredict
        - val val do_repair_update     = bpd_update_repair
        - 逻辑部分：在下一个周期，如果三者有一个有效，则设定io.bpdupdate
        - 逻辑部分：如果当前周期，do_commit_update有效，则bpd_ptr+1
            - io.enq.ready := RegNext(!full || do_commit_update)
            - 如果没满或者是满了但是刚有提交更新，会多出一个表项

        //重定向的设置
        - val redirect_idx = io.redirect.bits
        - val redirect_entry = ram(redirect_idx)
        - val redirect_new_entry = WireInit(redirect_entry)
        - 逻辑部分：当io.redirect.valid有效时
            - 更新enq_ptr为WrapInc(io.redirect.bits, num_entries)，应该就是删除了之后的表项
            - 如果io.brupdate.b2.mispredict为真，则更新redirect_new_entry的一些信息
            - 更新ras_update，ras_update_pc，ras_update_idx的信息
        - 逻辑部分：当上个周期的io.redirect.valid有效时
            - 更新prev的一些寄存器的信息，因为上个周期被重定向了
            - prev_entry := RegNext(redirect_new_entry)
            - prev_ghist := bpd_ghist
            - prev_pc    := bpd_pc
            - 在本周期将上个周期的redirect_new_entry的信息记录的ram中
            - ram(RegNext(io.redirect.bits)) := RegNext(redirect_new_entry)

        //通过io的读端口，读取ftq的表项信息和next的一些信息
*/

/* fetch-buffer.scala
    class FetchBufferResp extends BoomBundle
        - val uops = Vec(coreWidth, Valid(new MicroOp()))

    //- 包含一些uops，从fetchbundle送到fetchbuffer？
    class FetchBuffer extends BoomModule with HasBoomCoreParameters with HasBoomFrontendParameters
        - val io: IO
            - val enq: Flipped(Decoupled(new FetchBundle()))    //输入数据
                - io.enq.ready := do_enq
            - val deq: new DecoupledIO(new FetchBufferResp())   //输出数据
            - input: val clear: bool

        - val numEntries = numFetchBufferEntries, 16
        - val numRows = numEntries / coreWidth, 16/2

        //fetchbuffer的基本信息
        - val ram: Reg(Vec(numEntries, new MicroOp))
        - val deq_vec: Wire(Vec(numRows, Vec(coreWidth, new MicroOp)))
            - MicroOp[2][8]
        - val head = RegInit(1.U(numRows.W)), UInt<8>, clock
        - val tail = RegInit(1.U(numEntries.W))
        - val maybe_full = RegInit(false)

        //enqueue uops
        // Step 1: Convert FetchPacket into a vector of MicroOps.
        // Step 2: Generate one-hot write indices.
        // Step 3: Write MicroOps into the RAM.
        - def rotateLeft (in: UInt, k: Int)
            - 将in的低k为放到高位
        - val might_hit_head, at_head: 一些判断条件，但是不太懂
        - val do_enq: !(at_head && maybe_full || might_hit_head)
            - 判断当前是否能进入buffer
        
        - val in_mask: bool[fetchWidth]
        - val in_uops: MicroOp[fetchWidth]
        - 逻辑部分：
            - 将io.enq中取到的指令转变为一组MicroOPs放到in_uops中
                - for (b <- 0 until nBanks) {
                for (w <- 0 until bankWidth) {
            - 生成每个microOP的buffer索引，为了之后写入buffer中准备
                - val enq_idxs: uint<numEntries.w>[fetchWidth]
                - def inc(ptr: UInt): //循环左移一位
            - 将microOP写入到RAM中，根据do_enq && in_mask(i) && enq_idxs(i)(j)判断是否写入和写入的表项是哪一个
        
        //从buffer中弹出数据


        - io.deq.bits.uops zip Mux1H(head, deq_vec) map {case (d,q) => d.bits  := q}
            - //根据head选择那一个row用于deq，然后和uops组合成pair，然后利用map进行赋值
            - Mux1H, onehot选择信号
        
*/

/* frontend.scala
    //class BoomBundle(implicit val p: Parameters) extends freechips.rocketchip.util.ParameterizedBundle with HasBoomCoreParameters
    class FrontendResp extends BoomBundle
        - val pc = UInt(vaddrBitsExtended.W)  // ID stage PC
        - val data = UInt((fetchWidth * coreInstBits).W)
        - val mask = UInt(fetchWidth.W)
        - val xcpt = new FrontendExceptions
        - val ghist = new GlobalHistory

        - val fsrc = UInt(BSRC_SZ.W)  //BSRC_SZ: 2
        - val tsrc = UInt(BSRC_SZ.W)

    // used place: BoomCore & FrontendResp & GlobalHistory & FetchBundle & BoomFrontendIO & BoomFrontendModule & BranchPredictionUpdate & BranchPredictionRequest & GetPCFromFtqIO & FetchTargetQueue
    class GlobalHistory extends BoomBundle with HasBoomFrontendParameters
        //属性值
        - val old_history = UInt(globalHistoryLength.W)
        - val current_saw_branch_not_taken: bool
        - val new_saw_branch_not_taken, new_saw_branch_taken: bool
            - 是否新出现了分支不跳转/跳转的事件，
            - 如果都为false，意味着没有出现branch的任何实现，历史不用更新
        - val ras_idx: UInt(log2Ceil(nRasEntries).W)

        //定义的函数
        - def ===(other: globalhistory): //重载了比较是否相等的操作符
            - 相等的条件：
                - old_history相等
                - new_saw_branch_not_taken和new_saw_branch_taken也都相等
        - def =/=(): 不相等的比较符号

        - def histories(bank: Int): //获取当前最新的全局历史信息
            - 如果nBanks=1，则返回old_histroy
            - 否则要求nBanks至多为2
                - bank=0，返回old_history
                - bank=1，将new_saw_branch_not_taken和new_saw_branch_taken的信息写入old_history

        - def update(): globalhistory //返回全局历史
            - 一般情况下不进行更新，除非是发生了转移指令发生了taken，
                - 如果没有发生taken，则直接将未taken的信息记录在current_saw_branch_not_taken中即可，在下次update的时候该信息会加入进来
            - 函数参数：
                branches: UInt, //not clear, what is for ?，分支的数量？
                    - 转移预测器提供的预测信息
                    - 取指块中(是br&&predicted_pc.valid) & f1_mask
                cfi_idx: UInt, addr: UInt, 
                    - cfi_idx： 只有一位为1，f1_redirect_idx 
                cfi_taken: Bool, cfi_valid: Bool, 
                    - cfi_valid = f1_do_redirect
                    - cfi_taken = s1_bpd_resp.preds(f1_redirect_idx).taken && f1_do_redirect
                cfi_is_br: Bool, cfi_is_call: Bool, cfi_is_ret: Bool
            
            - val cfi_idx_fixed: 截断cfi_idx，获取有效数据
            - val cfi_idx_oh: cfi_idx的onehot形式
            - val new_histor = wire(new Globalhistory)  //逻辑电路的输出
            - val not_taken_branches: 应该是表明未发生跳转的分支指令数量，具体设计没搞懂
            
            - 逻辑部分：
                - nBanks == 1，此时bank能够看到之前的所有的历史信息
                    - new_history.current_saw_branch_not_taken := false.B：相当于重置，因为oldhistory会加入这些信息
                    - new_history.old_history： 根据update的信息更新当前的历史信息
                - nBanks == 2，此时bank无法看到上一次的信息，因为是对应于另一个bank
                    - val base = histories(1)
                    - val cfi_in_bank_0: cfi对应的指令是否在bank0中
                    - val ignore_second_bank: 是否忽略bank1，即在bank0中并且没有跨越bank
                    - val first_bank_saw_not_taken: 第一个bank是否能够看到分支没有跳转
                    - 重置new_history.current_saw_branch_not_taken := false.B
                    - 根据ignore_second_bank判断如何更新new_history中的信息
                - 更新ras_idx
                

    class HasBoomFrontendParameters extends HasL1ICacheParameters
        - val nBanks = if (cacheParams.fetchBytes <= 8) 1 else 2
            - 最大为2，根据取指宽度决定
        - val bankBytes = fetchBytes/nBanks
        - val bankWidth = fetchWidth(4/8)/nBanks 
        - val numChunks = cacheParams.blockBytes(64) / bankBytes
            - 一个cacheline被切分成多少个切片，交错排列
            - cacheParams.blockBytes/fetchBytes * nBanks
        
        - def bank(addr: Uint): 判断当前的地址属于哪一个bank
        - def isLastBankInBlock(addr: Uint): 判断是否是cacheline中的最后一个bank（nbank=2）
        - def mayNotBeDualBanked(addr: UInt): 猜测，表示地址不会跨越两个bank
            - isLastBankInBlock(addr)
        - def blockAlign(addr: UInt):  
        - def bankAlign(addr: UInt): 
            - 将地址按照block/bank进行对齐处理，即低若干位抹零
        
        - def fetchIdx(addr: UInt): 计算取指块的地址，右移若干位
        - def nextBank(addr: UInt): 
        - def nextFetch(addr: UInt): 需要判断当前是否为多bank的设计，以及判断是否达到bank的边界
        - def fetchMask(addr: UInt): 计算指令在整个取指块中的掩码
        - def bankMask(addr: UInt): 


    //used place: BoomFrontendModule & FetchBuffer & FetchTargetQueue
    //传递到fetchbuffer，主要用于组合多个相关的信号
    class FetchBundle extends BoomBundle with HasBoomFrontendParameters
        - val pc            = UInt(vaddrBitsExtended.W)
        - val next_pc       = UInt(vaddrBitsExtended.W)
        - val edge_inst     = Vec(nBanks, Bool())   
            - True if 1st instruction in this bundle is pc - 2， 官方解释，不太懂
        - val insts         = Vec(fetchWidth, Bits(32.W))
        - val exp_insts     = Vec(fetchWidth, Bits(32.W))

        // information for sfb folding, sfb:  short-forward branch (SFB) 
        - val sfbs                 = Vec(fetchWidth, Bool())
        - val sfb_masks            = Vec(fetchWidth, UInt((2*fetchWidth).W))
        - val sfb_dests            = Vec(fetchWidth, UInt((1+log2Ceil(fetchBytes)).W))
        - val shadowable_mask      = Vec(fetchWidth, Bool())
        - val shadowed_mask        = Vec(fetchWidth, Bool())

        // control flow instruction information
        - val cfi_idx       = Valid(UInt(log2Ceil(fetchWidth).W))
        - val cfi_type      = UInt(CFI_SZ.W), , jar/br/jmp之类的
        - val cfi_is_call   = Bool()
        - val cfi_is_ret    = Bool()
        - val cfi_npc_plus4 = Bool()
        - val br_mask   = UInt(fetchWidth.W)

        - val fsrc    = UInt(BSRC_SZ.W)
            - Source of the prediction from this bundle
        - val tsrc    = UInt(BSRC_SZ.W)
            - Source of the prediction to this bundle

        // ras information
        - val ras_top       = UInt(vaddrBitsExtended.W)

        //fetch target queue : fetch address and branch prediction snapshot
        - val ftq_idx       = UInt(log2Ceil(ftqSz).W)
        - val mask          = UInt(fetchWidth.W)    
            -  mark which words are valid instructions
        
        // history information
        - val ghist         = new GlobalHistory
        - val lhist         = Vec(nBanks, UInt(localHistoryLength.W))
        - val bpd_meta      = Vec(nBanks, UInt())

        // exception information
        - val xcpt_pf_if    = Bool() // I-TLB miss (instruction fetch fault).
        - val xcpt_ae_if    = Bool() // Access exception.

        // 不知道？
        - val bp_debug_if_oh= Vec(fetchWidth, Bool())  
        - val bp_xcpt_if_oh = Vec(fetchWidth, Bool())
        - val end_half      = Valid(UInt(16.W))

    //用于前端和CPU之间传递信息的IO
    class BoomFrontendIO extends BoomBundle
        - val fetchpacket = Flipped(new DecoupledIO(new FetchBufferResp))
            - 向后端传递指令包
        - val get_pc = Flipped(Vec(2, new GetPCFromFtqIO()))
            - 其中一个用于xcpt/jalr/auipc/flush，从ftq中获取恢复的信息
        - val debug_ftq_idx  = Output(Vec(coreWidth, UInt(log2Ceil(ftqSz).W)))
        - val debug_fetch_pc    = Input(Vec(coreWidth, UInt(vaddrBitsExtended.W)))

        - val status  = Output(new MStatus)
        - val bp    = Output(Vec(nBreakpoints, new BP))
            - 断点的信息，breakpoint
        
        //用于同步页表更新
        - val sfence = Valid(new SFenceReq)
        - val brupdate  = Output(new BrUpdateInfo)  //functional-unit.scala
            - 应该是用于在之后的阶段提供更新时需要的信息

        //用于更改pc时的一些变化信息
        - val redirect_flush   = Output(Bool()) // Flush and hang the frontend?
        - val redirect_val     = Output(Bool()) // Redirect the frontend?
        - val redirect_pc      = Output(UInt()) // Where do we redirect to?
        - val redirect_ftq_idx = Output(UInt()) // Which ftq entry should we reset to?
        - val redirect_ghist   = Output(new GlobalHistory) 
            - What are we setting as the global history?
        
        //尚不清楚
        - val commit = valid(UInt(ftqSz.W))
            - 提交时更新前端的一些结构，主要是ftq的表项
        - val flush_icache = Output(Bool())
            - 后端通知前端需要刷新icache
        - val perf = Input(new FrontendPerfEvents)

    class BoomFrontend (val icacheParams: ICacheParams, hartid: Int) extends LazyModule
        - hartid: id for the hardware thread of the core
        - lazy val module = new BoomFrontendModule(this)
        - val icache = LazyModule(new boom.ifu.ICache(icacheParams, hartid))
        - val masterNode = icache.masterNode


    //包裹整个前端的IO
    class BoomFrontendBundle(BoomFrontend) extends CoreBundle with HasExternallyDrivenTileConstants
        - val cpu : Flipped(BoomFrontendIO)
        - val ptw : TLBPTWIO
        - val errors = ICacheErrors

    //前端的主要模块，用于连接i-cache，tlb，取指控制器和转移预测流水线
    class BoomFrontendModule(outer: BoomFrontend) extends LazyModuleImp with HasBoomCoreParameters with HasBoomFrontendParameters
        - val io: IO(BoomFrontendBundle)
            - icache = outer.icache.module
                - icache.io.hartid     := io.hartid
                - icache.io.invalidate := io.cpu.flush_icache
                - icache.io.req.valid     := s0_valid
                - icache.io.req.bits.addr := s0_vpc
                - icache.io.s1_paddr := s1_ppc
                - icache.io.s1_kill  := tlb.io.resp.miss || f1_clear
                - icache.io.s2_kill := s2_xcpt
            - bpd = Module(new BranchPredictor)
                - bpd.io.f3_fire := false.B
                    when (f3_bpd_resp.io.enq.fire())
                        bpd.io.f3_fire := true.B
                - bpd.io.f0_req.valid      := s0_valid
                - bpd.io.f0_req.bits.pc    := s0_vpc
                - bpd.io.f0_req.bits.ghist := s0_ghist
                - bpd.io.update := bpd_update_arbiter.io.out
                - f3_bpd_resp.io.enq.bits  := bpd.io.resp.f3

            - ras = Module(new BoomRAS)
                - ras.io.read_idx := ras_read_idx
                    - when (f3.io.enq.fire())
                        ras_read_idx := f3.io.enq.bits.ghist.ras_idx
                        ras.io.read_idx := f3.io.enq.bits.ghist.ras_idx
                - ras.io.write_addr  := f3_aligned_pc + (f3_fetch_bundle.cfi_idx.bits << 1) + Mux(f3_fetch_bundle.cfi_npc_plus4, 4.U, 2.U)
                - ras.io.write_idx   := WrapInc(f3_fetch_bundle.ghist.ras_idx, nRasEntries)
                - ras.io.write_valid := false.B
                    - f3.io.deq.valid && f4_ready && f3_fetch_bundle.cfi_is_call && f3_fetch_bundle.cfi_idx.valid
                        ras.io.write_valid := true.B
                    - ftq.io.ras_update && enableRasTopRepair.B
                        ras.io.write_valid := true.B
                        ras.io.write_idx   := ftq.io.ras_update_idx
                        ras.io.write_addr  := ftq.io.ras_update_pc
            - tlb = Module()
                - io.ptw <> tlb.io.ptw 
                - tlb.io.req.valid      := (s1_valid && !s1_is_replay && !f1_clear) || s1_is_sfence
                - tlb.io.req.bits.cmd   := DontCare
                - tlb.io.req.bits.vaddr := s1_vpc
                - tlb.io.req.bits.passthrough := false.B
                - tlb.io.req.bits.size  := log2Ceil(coreInstBytes * fetchWidth).U
                - tlb.io.sfence         := RegNext(io.cpu.sfence)
                - tlb.io.kill           := false.B

            - io.cpu
                - io.cpu.perf.tlbMiss := io.ptw.req.fire()
                - io.cpu.perf.acquire := icache.io.perf.acquire
                - io.cpu.fetchpacket <> fb.io.deq
                - io.cpu.get_pc <> ftq.io.get_ftq_pc
                - io.cpu.debug_fetch_pc := ftq.io.debug_fetch_pc

            - val fb  = Module(new FetchBuffer)
                - fb.io.enq.valid := f4.io.deq.valid && ftq.io.enq.ready && !f4_delay
                - fb.io.enq.bits  := f4.io.deq.bits
                - fb.io.enq.bits.ftq_idx := ftq.io.enq_idx
                - fb.io.enq.bits.sfbs    := Mux(f4_sfb_valid, UIntToOH(f4_sfb_idx), 0.U(fetchWidth.W)).asBools
                - fb.io.enq.bits.shadowed_mask := (
                    Mux(f4_sfb_valid, f4_sfb_mask(fetchWidth-1,0), 0.U(fetchWidth.W)) |
                    f4.io.deq.bits.shadowed_mask.asUInt).asBools
                - fb.io.clear := false.B
                    - io.cpu.sfence.valid || io.cpu.redirect_flush
                         b.io.clear := true.B

            - val ftq = Module(new FetchTargetQueue)
                - ftq.io.enq.valid   := f4.io.deq.valid && fb.io.enq.ready && !f4_delay
                - ftq.io.enq.bits    := f4.io.deq.bits
                - ftq.io.deq := io.cpu.commit
                    - 异常/特殊指令/正常提交
                - ftq.io.brupdate := io.cpu.brupdate
                - ftq.io.redirect.valid   := io.cpu.redirect_val
                - ftq.io.redirect.bits    := io.cpu.redirect_ftq_idx
                    - io.cpu.redirect_flush: true
                        ftq.io.redirect.valid := io.cpu.redirect_val
                        ftq.io.redirect.bits  := io.cpu.redirect_ftq_idx
                - ftq.io.debug_ftq_idx := io.cpu.debug_ftq_idx


        - implicit val edge = outer.masterNode.edges.out(0)
            - 猜测是i-cache传递的信息接口
            - implicit声明的变量主要是用于在调用其它类的函数时，隐式的作为参数传递到那些函数中，从而如何设置被调参数的主动权就完全在主调函数一边

        //转移预测器
        - val bpd: Module(BranchPredictor)
            - bpd.io.f3_fire := false.B
        - val ras: Module(BoomRAS)

        //icache
        - val icache = outer.icache.module
            - icache.io.hartid     := io.hartid
            - icache.io.invalidate := io.cpu.flush_icache
                - 传递是否需要刷新i-cache
        //tlb, 应该是rocket定义的TLB class
        - val tlb: Module(new TLB(true, log2Ceil(fetchBytes), TLBConfig(nTLBEntries)))
           - io.ptw <> tlb.io.ptw   //接口互联，Flipped接口的应用， page table work
           - io.cpu.perf.tlbMiss := io.ptw.req.fire()
           - io.cpu.perf.acquire := icache.io.perf.acquire
            - perf看起来是为了统计一些执行信息使用


        // 流水线阶段1： 选择NextPC，F0，并且会发送请求到I-cache中
        - val s0_vpc       = WireInit(0.U(vaddrBitsExtended.W))
            - 虚拟地址pc
        - val s0_ghist     = WireInit((0.U).asTypeOf(new GlobalHistory))
            - s0阶段的ghist
        - val s0_tsrc      = WireInit(0.U(BSRC_SZ.W))
            - Source of the prediction to this bundle
            - tsrc出现的地方：FrontendResp， FetchBundle
            - BSRC_SZ：2， Which branch predictor predicted us
                - val BSRC_1 = 0.U(BSRC_SZ.W) // 1-cycle branch pred
                - val BSRC_2 = 1.U(BSRC_SZ.W) // 2-cycle branch pred
                - val BSRC_3 = 2.U(BSRC_SZ.W) // 3-cycle branch pred
                - val BSRC_C = 3.U(BSRC_SZ.W) // core branch resolution

        - val s0_valid     = WireInit(false.B)
        - val s0_is_replay = WireInit(false.B)
        - val s0_is_sfence = WireInit(false.B)
            - ���些判断信号，包括是否有效等，后两个尚不明确
            - 后两个直接传递到了s1中
        - val s0_replay_resp = Wire(new TLBResp)
            - 从TLB接收到重新执行的信息
            - 会被传递到s1中，用于在s1发现正在replay的情况下设置s1_tlb_resp
            - s0_replay_resp := s2_tlb_resp
        - val s0_replay_bpd_resp = Wire(new BranchPredictionBundle)
            - 转移预测器发出重新执行
            - s0_replay_bpd_resp := f2_bpd_resp
        - val s0_replay_ppc  = Wire(UInt())
            - ppc: paddr, pc
            - s0_replay_ppc  := s2_ppc

        - val s0_s1_use_f3_bpd_resp = WireInit(false.B)
            - 是否f3阶段的bpd的响应结果
        - 逻辑电路：
            - 如果本周期不需要reset，但是下周期需要reset，则重设一些信号
                - s0_valid   := true.B
                - s0_vpc     := io.reset_vector
                - s0_ghist   := (0.U).asTypeOf(new GlobalHistory)
                - s0_tsrc    := BSRC_C
            - 连线设置：
                - icache.io.req.valid     := s0_valid
                - icache.io.req.bits.addr := s0_vpc
                    - icache的请求信息
                - bpd.io.f0_req.valid      := s0_valid
                - bpd.io.f0_req.bits.pc    := s0_vpc
                - bpd.io.f0_req.bits.ghist := s0_ghist
                    - 发向转移预测器的请求，f0


        // 流水线阶段2： I-cache访问，f1，并且将vpc转换为ppc
            - val s1_vpc       = RegNext(s0_vpc)
            - val s1_ghist     = RegNext(s0_ghist)
            - val s1_tsrc      = RegNext(s0_tsrc)
            - val s1_valid     = RegNext(s0_valid, false.B)
            - val s1_is_replay = RegNext(s0_is_replay)
            - val s1_is_sfence = RegNext(s0_is_sfence)
                - 将s0中的信息继续保存到下一个阶段
                - s1_valid在reset时会被设置为false

            - val f1_clear     = WireInit(false.B)
                - 会在s2阶段设置，用于决定是否取消tlb和icache的访问请求

            - 逻辑部分1：设置tlb的访问请求信息
                - tlb.io.req.valid := (s1_valid && !s1_is_replay && !f1_clear) || s1_is_sfence
                    - 设定tlb的访问请求是否有效，fence
                - tlb.io.req.bits.cmd   := DontCare
                - tlb.io.req.bits.vaddr := s1_vpc
                - tlb.io.req.bits.passthrough := false.B        // ？似乎时为了虚拟化使用
                - tlb.io.req.bits.size  := log2Ceil(coreInstBytes * fetchWidth).U
                - tlb.io.sfence         := RegNext(io.cpu.sfence)
                - tlb.io.kill           := false.B

            //tlb的响应信息设置
            - val s1_tlb_miss = !s1_is_replay && tlb.io.resp.miss
            - val s1_tlb_resp = Mux(s1_is_replay, RegNext(s0_replay_resp), tlb.io.resp)
                - 如果当前s1正在replay，则设置为上个周期s0获取的replay_resp
                - s0_replay_resp := s2_tlb_resp
            - val s1_ppc  = Mux(s1_is_replay, RegNext(s0_replay_ppc), tlb.io.resp.paddr)
            
            //bpd在f1阶段的响应信息，一周期的预测
            - val s1_bpd_resp = bpd.io.resp.f1

            //设置i-cache中关于s1阶段的接口信息
            - icache.io.s1_paddr := s1_ppc
            - icache.io.s1_kill  := tlb.io.resp.miss || f1_clear
                - icache.io包括了s1_kill, s2_kill，因为需要两个周期才能够实际拿到指令？
                - 如果tlb发生了失效或者是f1周期需要被clear了，则立即终止icache的操作
                - s2_kill = s2_xcpt，即遇到异常时

            //f1阶段内部的一些信号
            - val f1_mask: 取指的掩码
            - val f1_redirects: bool[fetchwidth],根据预测的结果来判断是否需要重定向
            - val f1_redirect_idx: 需要重定向的指令在取指块中的哪一个位置
            - val f1_do_redirect: 是否需要重定向，f1_redirects异或（使用预测器的情况下）
            - val f1_targs: 根据预测器的结果，拿到所有预测的目标地址，使用f1_redirect_idx索引
            - val f1_predicted_target: 确定下一条取指的地址（预测的target/nextFetch）
            - val f1_predicted_ghist: s1_ghist.update
                - 根据当前的预测结果更新全局历史信息
                - 此时更新时一定是br更新，因为call/ret此时还不可知

            //逻辑部分2：如果s1_valid有效，并且没有发生tlb失效，则
                - 设置s0阶段用于选择pc的一些信息
                - s0_valid     := !(s1_tlb_resp.ae.inst || s1_tlb_resp.pf.inst)
                    - Stop fetching on fault
                    - access excpt, page fault
                - s0_tsrc      := BSRC_1
                - s0_vpc       := f1_predicted_target
                - s0_ghist     := f1_predicted_ghist
                - s0_is_replay := false.B

        // 流水线阶段3： I-cache访问请求的返回
            - val s2_valid = RegNext(s1_valid && !f1_clear, false.B)
                - s1有效，并且没有发生f1 clear
            - val s2_vpc = = RegNext(s1_vpc)
            - val s2_ghist = Reg(new GlobalHistory)
                - s2_ghist = s1_ghist
                - 即是直接相连过去的，而之前都是regnext，但是好像没有区别，都是晚一个周期
            - val s2_ppc  = RegNext(s1_ppc)
            - val s2_tsrc = RegNext(s1_tsrc)
                - provides the predictor component which provided the prediction TO this instruction
            - val s2_fsrc = WireInit(BSRC_1)
                - provides the predictor component which provided the prediction FROM this instruction
            - val s2_tlb_resp = RegNext(s1_tlb_resp)
            - val s2_tlb_miss = RegNext(s1_tlb_miss)
            - val s2_is_replay = RegNext(s1_is_replay) && s2_valid
                - 都是从上个周期的s1中继续保存信息

            - val f2_clear = WireInit(false.B)
            - val s2_xcpt = s2_valid && (s2_tlb_resp.ae.inst || s2_tlb_resp.pf.inst) && !s2_is_replay
                - 用于指明当前s2阶段是否存在异常需要处理
                - icache.io.s2_kill := s2_xcpt
            - val f3_ready = Wire(Bool()) 
                - f3是将指令放入队列中，应该是用于判断s2的指令是否可以放入队列使用
                - f3_ready := f3.io.enq.ready
            
            // f2阶段内部的一些信号， 类似于f1阶段的那些信号，主要是关于转移预测的信息
            - val f2_bpd_resp = bpd.io.resp.f2
            - val f2_mask = fetchMask(s2_vpc)
            - val f2_redirects： bool[fetchWidth], 根据f2_bpd_resp判断预测器是否预测跳转
            - val f2_redirect_idx: 发生redirect的指令在fetch包中的哪一条指令
            - val f2_targs： 含义和f1阶段的同名信号一样
            - val f2_do_redirect， f2_predicted_target， f2_predicted_ghist
                - f2_predicted_ghist = s2_ghist.update

            - val f2_correct_f1_ghist： 判断是否需要纠正s1阶段预测器提供的结果
                - 如果支持enableGHistStallRepair，并且发现s1_ghist和f2_predicted_ghist不相同
                - s1_ghist = RegNext(s0_ghist)
                - f1_predicted_ghist: s1_ghist.update
                - s0_ghist := f1_predicted_ghist
                - f2_predicted_ghist = s2_ghist.update
                - s2_ghist = s1_ghist
                - 都是基于同一个s0然后和本周期的预测信息而更新的历史信息，然后进行比较
            - 逻辑部分：更新s0的一些信息
                - 当i-cache没有拿到正确的信息 或者 f3的ftq没有就绪
                    - 更新s0阶段的一些信号
                    - s0_is_replay := s2_valid && icache.io.resp.valid
                        - 即如果f3没有就绪，则f1需要清除，重新访问i-cache，f1会重新执行
                        - 如果请求失效，则从s0重新执行
                    - s0_s1_use_f3_bpd_resp := !s2_is_replay，如果不replay，则使用f3的预测器信息。（这个信号没有找到哪里用了）
                    - f1_clear = true.B
                        - icache.io.s1_kill  := tlb.io.resp.miss || f1_clear
                        - s2_valid = RegNext(s1_valid && !f1_clear, false.B)
                - 当f3_ready时
                    - 如果s1_valid && s1_vpc === f2_predicted_target && !f2_correct_f1_ghist
                        - 更新s2_ghist, s2_ghist := f2_predicted_ghist
                    - 如果s1_vpc =/= f2_predicted_target || f2_correct_f1_ghist ||!s1_valid
                        - 预测错了或者s1是无效的则，f1_clear := true.B，清除f1，重新开始
                        - s2_fsrc | s0_tsrc := BSRC_2， s2提供了预测
                        - s0_ghist := f2_predicted_ghist， s0_vpc := f2_predicted_target
                            - 根据s2的预测结果重新执行
                - 设置一些需要replay时的信息，包括
                    - s0_replay_bpd_resp := f2_bpd_resp
                    - s0_replay_resp := s2_tlb_resp
                    - s0_replay_ppc  := s2_ppc                      
            
        // 流水线阶段4： f3，Instruction Fetch 3 (enqueue to fetch buffer)
            - val f3_clear = WireInit(false.B)
            - val f4_ready = bool
            //新的reset域中的两个变量：withReset(reset.toBool || f3_clear)
                - f3 = Module(new Queue(new FrontendResp, 1, pipe=true, flow=false)) 
                - f3_bpd_resp: new Queue(new BranchPredictionBundle, 1, pipe=true, flow=true))
            - 逻辑部分：f3.io.enq的设置
                - valid: 何时将产生的FrontendResp放入队列
                    - s2_valid && !f2_clear： s2的信息有效
                    - 并且，icache.io.resp.valid || ((s2_tlb_resp.ae.inst || s2_tlb_resp.pf.inst： 出现了这三种情况
                    - 并且，!s2_tlb_miss （这种情况不做任何通知？）
                - FrontendResp的元素信息： 主要是s2阶段返回的信息
                    - f3.io.enq.bits.pc := s2_vpc
                    - f3.io.enq.bits.data  := Mux(s2_xcpt, 0.U, icache.io.resp.bits.data)
                    - f3.io.enq.bits.ghist := s2_ghist
                    - f3.io.enq.bits.mask := fetchMask(s2_vpc)
                    - f3.io.enq.bits.xcpt := s2_tlb_resp
                    - f3.io.enq.bits.fsrc := s2_fsrc
                    - f3.io.enq.bits.tsrc := s2_tsrc
                - 当FrontendResp进入队列时，即f3.io.enq.fire()（ready && valid）
                    - 将Resp中记录的ras_idx赋值给ras和一个内部寄存器ras_read_idx
                    - ras_read_idx := f3.io.enq.bits.ghist.ras_idx
                    - ras.io.read_idx := f3.io.enq.bits.ghist.ras_idx
                    - ras的读取需要一个周期，所以需要等到下个周期才行
                    - 这个周期应该可以确定指令是否时ret指令？f2是不是也可以了？
            - 逻辑部分：f3_bpd_resp.io.enq设置
                - valid: f3.io.deq.valid && RegNext(f3.io.enq.ready)
                    - 要求f3当前可以出队列，并且上个周期刚刚进入队列（ready意味着是空的）
                    - 感觉意思应该是说上个周期刚刚有FrontendResp进入了队列
                    - io.deq.valid := !empty
                    - io.enq.ready := !full
                - bits: bpd.io.resp.f3 (BranchPredictionBundle)
                    - 如果队列允许压入元素，则需要告知bpd，即bpd.io.f3_fire := true.B
                    - f3_fire在预测器中应该是用于判断当前能够更新预测器
                    - if(f3_fire) f3_do_update=true; (local.scala中涉及到)
            - 逻辑部分：f3_bpd_resp.io.deq和f3.io.deq的信号设置
                - f3.io.deq.ready := f4_ready
                - f3_bpd_resp.io.deq.ready := f4_ready
                    - f4_ready := f4.io.enq.ready，即只有当f4的队列插入就绪时才能够deq

            - f3阶段的内部信号：
                - val f3_imemresp     = f3.io.deq.bits
                    - 从i-cache返回的数据和s2阶段的一些信息
                - val f3_bank_mask    = bankMask(f3_imemresp.pc)
                - val f3_data         = f3_imemresp.data
                    - i-cache返回的数据
                - val f3_aligned_pc   = bankAlign(f3_imemresp.pc)
                    - 计算对齐之后的pc地址
                - val f3_is_last_bank_in_block = isLastBankInBlock(f3_aligned_pc)
                    - 当前是否是block中的最后一个bank，是否跨越了block？
                - val f3_is_rvc       = Wire(Vec(fetchWidth, Bool()))
                    - isRVC(inst: UInt) = (inst(1,0) =/= 3.U)
                    - 记录当前取值的每16位是不是一条rvc
                - val f3_redirects    = Wire(Vec(fetchWidth, Bool()))
                - val f3_targs        = Wire(Vec(fetchWidth, UInt(vaddrBitsExtended.W)))
                - val f3_cfi_types    = Wire(Vec(fetchWidth, UInt(CFI_SZ.W)))
                - val f3_shadowed_mask = Wire(Vec(fetchWidth, Bool()))
                - val f3_mask         = Wire(Vec(fetchWidth, Bool()))
                - val f3_br_mask      = Wire(Vec(fetchWidth, Bool()))
                - val f3_call_mask    = Wire(Vec(fetchWidth, Bool()))
                - val f3_ret_mask     = Wire(Vec(fetchWidth, Bool()))
                - val f3_npc_plus4_mask = Wire(Vec(fetchWidth, Bool()))
                - val f3_btb_mispredicts = Wire(Vec(fetchWidth, Bool()))
                - val f3_fetch_bundle = Wire(new FetchBundle)   
                    //传递到fetchbuffer，主要用于组合多个相关的信号
                    - f3_fetch_bundle.mask := f3_mask.asUInt
                    - f3_fetch_bundle.br_mask := f3_br_mask.asUInt
                    - f3_fetch_bundle.pc := f3_imemresp.pc
                    - f3_fetch_bundle.ftq_idx := 0.U // This gets assigned later
                    - f3_fetch_bundle.xcpt_pf_if := f3_imemresp.xcpt.pf.inst
                    - f3_fetch_bundle.xcpt_ae_if := f3_imemresp.xcpt.ae.inst
                    - f3_fetch_bundle.fsrc := f3_imemresp.fsrc
                    - f3_fetch_bundle.tsrc := f3_imemresp.tsrc
                    - f3_fetch_bundle.shadowed_mask := f3_shadowed_mask


                - def isRVC(inst: Int) = (inst(1,0) =/= 3.U)
                    - 根据指令低两位判断是否为压缩指令
                    - The optional compressed 16-bit instruction-set extensions have their lowest two bits equal to 00, 01, or 10.

                - val f3_prev_half    = Reg(UInt(16.W))
                    - 追踪前一个获取数据包的尾部16b（Tracks trailing 16b of previous fetch packet）
                - val f3_prev_is_half = RegInit(false.B)
                    - 记录上个取指包中是否包含了half-inst
                - var bank_prev_is_half = f3_prev_is_half
                - var bank_prev_half    = f3_prev_half
                    - var变量也出现在了最后的verilog代码中
                    - 变成了wire类型，而f3_prev_is_half成了reg
                    - when (f3.io.deq.fire()) {
                        f3_prev_is_half := bank_prev_is_half
                        f3_prev_half    := bank_prev_half
                        assert(f3_bpd_resp.io.deq.bits.pc === f3_fetch_bundle.pc)
                    - when (f3_clear) f3_prev_is_half = false
                - var redirect_found = false
                - var last_inst : 16bits

                //逻辑部分1：
                - 从f3_data(上一个阶段i-cache读取的信息)中获取数据，以bank循环来获取
                    - val bank_data, bank_mask, bank_insts(vec(bankwidth, 32bits))
                - 在bankwidth中循环，挨个获取一条指令
                    - if w == 0, bank中第一块的情况，需要考虑是否有32位指令的一半在上一个bank中
                        - val inst0 = Cat(bank_data(15,0), f3_prev_half)
                            - 将上一次遗留下来的一半指令拿过来使用
                        - val inst1 = bank_data(31,0)
                            - 当前bank的32位。虽然不确定是否是rvc，但是先获取32位，
                            - 如果是rvc，则只会根据低16位扩展得到32位，否则会正常拿到32位指令
                        - val pc0, pc1: inst0和inst1两种情况下的pc地址
                        - bpd_decoder0.io.inst := exp_inst0， bpd_decoder0.io.pc   := pc0
                            - 为两条指令指定译码，判断是否为branch
                            - 译码单元返回的信号：BranchDecodeSignals
                                - is_ret, is_call, target, cfi_type, sfb_offset, shadowable
                        - 根据bank_prev_is_half判断是否在上一个bank遗留了一半的指令
                            - 如果是，则将inst0的信息放入到f3_fetch_bundle中
                                - f3_fetch_bundle.edge_inst(b) := true.B，即bank的边界指令
                                - 特殊情况：即当前的bank处于取指块的第一个，则使用f3_prev_half，否则需要使用last_inst来组合得到新的指令
                            - 否则，将inst1的信息放入f3_fetch_bundle中
                    
                    - w!=0的情况，16位指令进行判断
                        - 获取指令，判断是否是rvc，得到32指令，并且进行branch的预译码
                        - 如果w=1，inst为获取bank_data的47-16位
                            - valid信号用于表示当前w获取的指令是否有效
                            - 只有当上一次拿到的是16位指令时（或者类似情况），valid才是有效的
                        - 如果w==bankwidth-1，即最后一个
                            - inst的高16位拼接为0，地位为bank的最后16位，
                            - 此时只有当前16位是rvc，并且当前也是rvc才会使得valid有效
                        - 其它情况
                            - 获取w*16开始的32位设为inst，并且设定valid，
                            - valid=true, 上个w没有有效指令，并且也不是rvc指令
                    
                    //统一的设置
                    - 更新f3_is_rvc：每隔16位设定标志位，即是否当前16位是一条rvc
                        - f3_is_rvc(i) := isRVC(bank_insts(w))
                        - i = (b * bankWidth) + w
                    - 设置bank_mask(w): f3deq有效(即到此时的数据是有效的)&&取值mask有效&&指令译码有效&&没有发现重定向(!redirect_found)
                    - 设置f3_mask(i)：和bank_mask一样，前者主要用来解析取值块中的指令
                        - 如果之前遇到的指令发现会跳转，则之后的指令都可以设为无效

                    - 更新f3_targs(i)：根据指令预译码的结果设置targets
                        - 如果是jalr，则使用预测的目标，否则使用译码的目标
                    - 设置f3_btb_mispredicts(i)：如果是jal指令，则预译码之后target可以确定，否则发现和预测的target不一致，则需要更新btb的表项，并且通知预测错误
                    - 设置f3_npc_plus4_mask(i): 
                        - 用于计算取值块中每条指令的pc，因此需要判断指令是否是一条rvc
                    - 设置sfb的一些参数，包括
                        - f3_fetch_bundle.sfbs(i)
                        - f3_fetch_bundle.sfb_masks(i)
                        - f3_fetch_bundle.shadowable_mask(i)
                        - f3_fetch_bundle.sfb_dests(i)

                    - 设置f3_redirects(i)，即是否会发生跳转
                        - 指令有效，并且指令是jal/jalr,或者是br但是预测跳转
                    - 设置f3_br_mask(i)：分支指令的掩码，不包括jal/jalr
                    - 根据预译码结果设置f3_cfi_types(i), f3_call_mask(i), f3_ret_mask(i)
                    - redirect_found = redirect_found || f3_redirects(i)
                        - 当前取指块是否有重定向的可能

                    //尚不清楚
                    - f3_fetch_bundle.bp_debug_if_oh(i) := bpu.io.debug_if
                    - f3_fetch_bundle.bp_xcpt_if_oh (i) := bpu.io.xcpt_if
                
                 //bank循环的统一设置
                - last_inst = bank_insts(bankWidth-1)(15,0)
                    - 上一个bank的最后16位
                - 更新bank_prev_is_half和bank_prev_half的信息
                    - 如果f3_bank_mask(b)==0, 则猜测意味着当前bank没有取，因此沿用之前的信息
                    - 否则，使用当前bank最后16位的信息
            
            //遍历完成f3_data，并且填充f3_fetch_bundle后的逻辑部分
            - 设置f3_fetch_bundle的一些整体信息
                - cfi_type: f3_cfi_types
                - cfi_is_call: f3_call_mask
                - cfi_is_ret: f3_ret_mask
                - cfi_npc_plus4: f3_npc_plus4_mask
                - ghist    := f3.io.deq.bits.ghist
                - lhist    := f3_bpd_resp.io.deq.bits.lhist
                - bpd_meta := f3_bpd_resp.io.deq.bits.meta
                    - f3_bpd_resp.io.enq.bits  := bpd.io.resp.f3
                - end_half.valid := bank_prev_is_half
                - end_half.bits  := bank_prev_half
                - cfi_idx.valid := f3_redirects.reduce(_||_)
                - cfi_idx.bits  := PriorityEncoder(f3_redirects)
                    - 选择遇到的第一个
                - ras_top := ras.io.read_addr
                - next_pc := f3_predicted_target

            - 获取f3阶段转移预测的信息，然后指导前端取指
                - val f3_predicted_target: 预测的地址
                    - 如果是ret，则使用ras.io.read_addr
                    - 否则使用f3_targs(PriorityEncoder(f3_redirects))
                    - 如果不跳转，则使用pc+fetchwidth
                - val f3_predicted_ghist = f3_fetch_bundle.ghist.update()
                    - 根据f3_fetch_bundle的信息，更新全局历史，记录下来
                - 设置ras的更新信息
                    - ras.io.write_addr = pc+4/pc+2, 根据是否为rvc
                    - ras.io.write_idx = WrapInc(f3_fetch_bundle.ghist.ras_idx, nRasEntries)
                    - ras.io.write_valid = true if f3.io.deq.valid && f4_ready
                        - f3_fetch_bundle.cfi_is_call && f3_fetch_bundle.cfi_idx.valid
                - 判断当前是否需要重定向取指
                    - val f3_correct_f1_ghist = s1_ghist =/= f3_predicted_ghist
                    - val f3_correct_f2_ghist = s2_ghist =/= f3_predicted_ghist
                - 更新预测器，重定向，时机：f3.io.deq.valid && f4_ready
                    - 如果需要重定向（f3_redirects），则删除f3_prev_is_half
                    - 如果s2有效，并且s2的pc和f3预测的一样，并且！f3_correct_f2_ghist
                        - 则将当前f3更新之后的历史，传递到下一个周期的f3继续使用
                        - f3.io.enq.bits.ghist := f3_predicted_ghist
                    - 如果s2无效，s1有效，同时s1的取指地址和预测一致，并且!f3_correct_f1_ghist
                        - 则继续执行，并且将f3更新之后的历史设定为下一个周期的s2的ghist
                        - s2_ghist := f3_predicted_ghist
                    - 如果s2有效，但是(s2的pc和f3预测的不一样或者f3_correct_f2_ghist)
                     或 如果s2无效，s1有效, 但是(s1的pc和f3预测的不一样或者f3_correct_f1_ghist)
                     或 s1和s2都无效
                        - 则清除f1和f2，f2_clear=true；
                        - 重置s0的信息，进行取指
                        - s0_valid     := !(f3_fetch_bundle.xcpt_pf_if || f3_fetch_bundle.xcpt_ae_if)
                        - s0_vpc       := f3_predicted_target
                        - s0_is_replay := false.B
                        - s0_ghist     := f3_predicted_ghist
                        - s0_tsrc      := BSRC_3
                        - f3_fetch_bundle.fsrc := BSRC_3，记录在当前的取指包中
                - 如果f3阶段发现由于jal等指令导致BTB预测错误，则将该请求缓存在BranchPredictionUpdate的队列中，稍后处理
                    - val f4_btb_corrections = Module(Queue(BranchPredictionUpdate))
                    - f4_btb_corrections.io.enq.valid: f3出队列有效，并且发现了btb预测错误
                    - f4_btb_corrections.io.enq.bits  := DontCare
                    - f4_btb_corrections.io.enq.bits.is_mispredict_update := false.B
                    - f4_btb_corrections.io.enq.bits.is_repair_update     := false.B
                    - f4_btb_corrections.io.enq.bits.btb_mispredicts  := f3_btb_mispredicts.asUInt      （onehot）
                    - f4_btb_corrections.io.enq.bits.pc    := f3_fetch_bundle.pc
                    - f4_btb_corrections.io.enq.bits.ghist := f3_fetch_bundle.ghist
                    - f4_btb_corrections.io.enq.bits.lhist := f3_fetch_bundle.lhist
                    - f4_btb_corrections.io.enq.bits.meta  := f3_fetch_bundle.bpd_meta

        // 流水线阶段5： f4，Instruction Fetch 4 (redirect from bpd)
            - val f4_clear
            // 承载f3处理后的fetchbundle
            - val f4: withReset(reset.toBool || f4_clear)
                - queue(fetchBundle, 1, pipe=true, flow=false)
                - f4.io.enq.valid := f3.io.deq.valid && !f3_clear
                - f4.io.enq.bits  := f3_fetch_bundle
                
                //要求出队列时，另外两个队列都ready
                - f4.io.deq.ready := fb.io.enq.ready && ftq.io.enq.ready && !f4_delay
            - val f4_ready = f4.io.enq.ready: 用于判断当前f4是否可以插入

            //当f4出队列的数据有效时，并且ftq和fb都可以进入队列，则进入队列
            - val fb: fetchbuffer
                - fb.io.enq.valid := f4.io.deq.valid && ftq.io.enq.ready && !f4_delay
                    - 为了获得ftq的索引
                - fb.io.enq.bits  := f4.io.deq.bits
                - fb.io.enq.bits.ftq_idx := ftq.io.enq_idx
                - fb.io.enq.bits.sfbs    := Mux(f4_sfb_valid, UIntToOH(f4_sfb_idx), 0.U(fetchWidth.W)).asBools
                - fb.io.enq.bits.shadowed_mask

            //Queue used to track the branch prediction information for inflight Micro-Ops. This is dequeued once all instructions in its Fetch Packet entry are committed.
            //ftq存储的时候是按fetchbundle存储的，因此和rob的数量关系应该*fetchwidth进行比较
            - val ftq: fetchTargetQueue
                - ftq.io.enq.valid := f4.io.deq.valid && fb.io.enq.ready && !f4_delay
                - ftq.io.enq.bits  := f4.io.deq.bits
                
            - 处理sfbs的操作，目前没有细看

            - 根据ftq中的数据更新ras


        // **** To Core (F5) ****
        - 对接接口：cpu的接口和ftq, fb的接口
        - 根据cpu的事件，给出前端的一些处理
            - 遇到sfence事件
            - 遇到redirect_flush事件

*/