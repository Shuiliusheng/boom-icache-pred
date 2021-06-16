package boom.ifu

import chisel3._
import chisel3.util._

import freechips.rocketchip.config.{Field, Parameters}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._

import boom.common._
import boom.util.{BoomCoreStringPrefix, WrapInc}

import scala.math.min

case class BoomFAMicroBTBParams(
  nWays: Int = 16,
  offsetSz: Int = 13
)

//全相联BTB
class FAMicroBTBBranchPredictorBank(params: BoomFAMicroBTBParams = BoomFAMicroBTBParams())(implicit p: Parameters) extends BranchPredictorBank()(p)
{
  override val nWays         = params.nWays
  val tagSz         = vaddrBitsExtended - log2Ceil(fetchWidth) - 1
  val offsetSz      = params.offsetSz
  val nWrBypassEntries = 2

  def bimWrite(v: UInt, taken: Bool): UInt = {
    val old_bim_sat_taken  = v === 3.U
    val old_bim_sat_ntaken = v === 0.U
    Mux(old_bim_sat_taken  &&  taken, 3.U,
      Mux(old_bim_sat_ntaken && !taken, 0.U,
      Mux(taken, v + 1.U, v - 1.U)))
  }

  require(isPow2(nWays))

  class MicroBTBEntry extends Bundle {
    val offset   = SInt(offsetSz.W)
  }

  class MicroBTBMeta extends Bundle {
    val is_br = Bool()
    val tag   = UInt(tagSz.W)
    val ctr   = UInt(2.W)
  }

  class MicroBTBPredictMeta extends Bundle {
    val hits  = Vec(bankWidth, Bool())
    val write_way = UInt(log2Ceil(nWays).W)
  }

  class IndexInfo extends Bundle {
    val pc = UInt(48.W)
    val way = UInt(log2Ceil(nWays).W)
    val valid = Bool()
  }

  val s1_meta = Wire(new MicroBTBPredictMeta)
  override val metaSz = s1_meta.asUInt.getWidth


  val meta     = RegInit((0.U).asTypeOf(Vec(nWays, Vec(bankWidth, new MicroBTBMeta))))
  val btb      = Reg(Vec(nWays, Vec(bankWidth, new MicroBTBEntry)))
  
  //chw
  val pbits        = Reg(Vec(nWays, Vec(bankWidth, UInt(2.W))))
  val pbits_valid  = RegInit((0.U).asTypeOf(Vec(nWays, Vec(bankWidth, UInt(1.W)))))

  val hitinfo1 = RegInit((0.U).asTypeOf(new IndexInfo()))
  val hitinfo2 = RegNext(hitinfo1)
  val hitinfo3 = RegNext(hitinfo2)


  val mems = Nil
  val s1_req_tag   = s1_idx

  val s1_resp   = Wire(Vec(bankWidth, Valid(UInt(vaddrBitsExtended.W))))
  val s1_taken  = Wire(Vec(bankWidth, Bool()))
  val s1_is_br  = Wire(Vec(bankWidth, Bool()))
  val s1_is_jal = Wire(Vec(bankWidth, Bool()))

  val s1_hit_ohs = VecInit((0 until bankWidth) map { i =>
    VecInit((0 until nWays) map { w =>
      meta(w)(i).tag === s1_req_tag(tagSz-1,0)
    })
  })
  val s1_hits     = s1_hit_ohs.map { oh => oh.reduce(_||_) }
  val s1_hit_ways = s1_hit_ohs.map { oh => PriorityEncoder(oh) }

  for (w <- 0 until bankWidth) {
    val entry_meta = meta(s1_hit_ways(w))(w)
    s1_resp(w).valid := s1_valid && s1_hits(w)
    s1_resp(w).bits  := (s1_pc.asSInt + (w << 1).S + btb(s1_hit_ways(w))(w).offset).asUInt
    s1_is_br(w)      := s1_resp(w).valid &&  entry_meta.is_br
    s1_is_jal(w)     := s1_resp(w).valid && !entry_meta.is_br
    s1_taken(w)      := !entry_meta.is_br || entry_meta.ctr(1)

    s1_meta.hits(w)     := s1_hits(w)
  }
  val alloc_way = {
    val r_metas = Cat(VecInit(meta.map(e => VecInit(e.map(_.tag)))).asUInt, s1_idx(tagSz-1,0))
    val l = log2Ceil(nWays)
    val nChunks = (r_metas.getWidth + l - 1) / l
    val chunks = (0 until nChunks) map { i =>
      r_metas(min((i+1)*l, r_metas.getWidth)-1, i*l)
    }
    chunks.reduce(_^_)
  }
  s1_meta.write_way := Mux(s1_hits.reduce(_||_),
    PriorityEncoder(s1_hit_ohs.map(_.asUInt).reduce(_|_)),
    alloc_way)

  val debug_cycles = freechips.rocketchip.util.WideCounter(32)
  for (w <- 0 until bankWidth) {
    io.resp.f1(w).predicted_pc := s1_resp(w)
    io.resp.f1(w).is_br        := s1_is_br(w)
    io.resp.f1(w).is_jal       := s1_is_jal(w)
    io.resp.f1(w).taken        := s1_taken(w)

    //when(s1_hits(w)){
      // printf("faubtb s1_resp, cycle: %d, w: %d, valid: %d, hit: %d, pc: 0x%x, target pc: 0x%x\n", debug_cycles.value, w.U, s1_resp(w).valid, s1_hits(w), s1_idx << 3.U, s1_resp(w).bits)
    //}

    io.resp.f2(w) := RegNext(io.resp.f1(w))
    io.resp.f3(w) := RegNext(io.resp.f2(w))

    // printf("faubtb f3, cycle: %d, valid: %d, target pc: 0x%x\n", debug_cycles.value, io.resp.f3(w).predicted_pc.valid, io.resp.f3(w).predicted_pc.bits)
    // printf("faubtb f2, cycle: %d, valid: %d, target pc: 0x%x\n", debug_cycles.value, io.resp.f2(w).predicted_pc.valid, io.resp.f2(w).predicted_pc.bits)
    // printf("faubtb f1, cycle: %d, valid: %d, target pc: 0x%x\n", debug_cycles.value, io.resp.f1(w).predicted_pc.valid, io.resp.f1(w).predicted_pc.bits)
  }
  io.f3_meta := RegNext(RegNext(s1_meta.asUInt))

  val s1_update_cfi_idx = s1_update.bits.cfi_idx.bits
  val s1_update_meta    = s1_update.bits.meta.asTypeOf(new MicroBTBPredictMeta)
  val s1_update_write_way = s1_update_meta.write_way

  val max_offset_value = (~(0.U)((offsetSz-1).W)).asSInt
  val min_offset_value = Cat(1.B, (0.U)((offsetSz-1).W)).asSInt
  val new_offset_value = (s1_update.bits.target.asSInt -
    (s1_update.bits.pc + (s1_update.bits.cfi_idx.bits << 1)).asSInt)

  val s1_update_wbtb_data     = Wire(new MicroBTBEntry)
  s1_update_wbtb_data.offset := new_offset_value
  val s1_update_wbtb_mask = (UIntToOH(s1_update_cfi_idx) &
    Fill(bankWidth, s1_update.bits.cfi_idx.valid && s1_update.valid && s1_update.bits.cfi_taken && s1_update.bits.is_commit_update))

  val s1_update_wmeta_mask = ((s1_update_wbtb_mask | s1_update.bits.br_mask) &
    Fill(bankWidth, s1_update.valid && s1_update.bits.is_commit_update))

  // Write the BTB with the target
  when (s1_update.valid && s1_update.bits.cfi_taken && s1_update.bits.cfi_idx.valid && s1_update.bits.is_commit_update) {
    btb(s1_update_write_way)(s1_update_cfi_idx).offset := new_offset_value
    
    //chw: 对于刚更新的btb表项，首先将其标记为not valid,即没有正确的pbits预测值
    pbits_valid(s1_update_write_way)(s1_update_cfi_idx) := 0.U
  }

  // Write the meta
  for (w <- 0 until bankWidth) {
    when (s1_update.valid && s1_update.bits.is_commit_update &&
      (s1_update.bits.br_mask(w) ||
        (s1_update_cfi_idx === w.U && s1_update.bits.cfi_taken && s1_update.bits.cfi_idx.valid))) {
      val was_taken = (s1_update_cfi_idx === w.U && s1_update.bits.cfi_idx.valid &&
        (s1_update.bits.cfi_taken || s1_update.bits.cfi_is_jal))

      meta(s1_update_write_way)(w).is_br := s1_update.bits.br_mask(w)
      meta(s1_update_write_way)(w).tag   := s1_update_idx
      meta(s1_update_write_way)(w).ctr   := Mux(!s1_update_meta.hits(w),
        Mux(was_taken, 3.U, 0.U),
        bimWrite(meta(s1_update_write_way)(w).ctr, was_taken)
      )
    }
  }

  //////////////////////////////////////////////////////////////////////////
  when(s1_hits.reduce(_||_)){
    hitinfo1.way := s1_meta.write_way
    hitinfo1.valid := true.B
    hitinfo1.pc := s1_idx
    // printf("faubtb s1 hit, cycle: %d, way: %d, pc: 0x%x\n", debug_cycles.value, s1_meta.write_way, s1_idx << 3.U)
    //printf("faubtb meta cycle: %d, tag: 0x%x, 0x%x\n", debug_cycles.value, meta(s1_hit_ways(0))(0).tag, meta(s1_hit_ways(1))(1).tag)
  }
  .otherwise{
    hitinfo1.valid := false.B
  }

  //chw: update pred_bits info
  for (w <- 0 until bankWidth) {
    io.resp.f1_pbits(w).valid := Mux(pbits_valid(s1_hit_ways(w))(w) === 1.U, true.B, false.B)
    io.resp.f1_pbits(w).bits := pbits(s1_hit_ways(w))(w)
    // when(s1_hits(w)){
    // printf("faubtb pbits, cycle: %d, pc: 0x%x, w: %d, hit: %d, pbit_valid: %d, bits: %d\n", debug_cycles.value, s1_idx << 3.U, w.U, s1_hits(w), io.resp.f1_pbits(w).valid, io.resp.f1_pbits(w).bits)
    // }
  }

  //chw: update fx_hit_info
  io.resp.f2_pbits := RegNext(io.resp.f1_pbits)
  io.resp.f3_pbits := RegNext(io.resp.f2_pbits)


  //chw: 在地址转换完成之后，更新对应btb表项中的pred位
  val update_pc = fetchIdx(io.pbits_update.bits.pc)
  // when(io.pbits_update.valid){
  //   printf("faubtb update pbits, cycle: %d, update pc: 0x%x, | h1 valid: %d, pc: 0x%x | h2 valid: %d, pc: 0x%x | h3 valid: %d, pc: 0x%x\n", debug_cycles.value,
  //     update_pc << 3.U, hitinfo1.valid, hitinfo1.pc << 3.U, hitinfo2.valid, hitinfo2.pc << 3.U, hitinfo3.valid, hitinfo3.pc << 3.U)
  // }
  when(io.pbits_update.valid && (update_pc === hitinfo1.pc) && hitinfo1.valid){//有效，并且需要更新faubtb
    pbits_valid(hitinfo1.way)(io.pbits_update.bits.cfi_idx) := 1.U
    pbits(hitinfo1.way)(io.pbits_update.bits.cfi_idx) := io.pbits_update.bits.pbits

    // printf("faubtb update pbits in hit1, cycle: %d, pc: 0x%x, way: %d, cif_idx: %d, pbits: %d\n", debug_cycles.value,
    //   update_pc << 3.U, hitinfo1.way, io.pbits_update.bits.cfi_idx, io.pbits_update.bits.pbits)
  }
  .elsewhen(io.pbits_update.valid && (update_pc === hitinfo2.pc) && hitinfo2.valid){//有效，并且需要更新faubtb
    pbits_valid(hitinfo2.way)(io.pbits_update.bits.cfi_idx) := 1.U
    pbits(hitinfo2.way)(io.pbits_update.bits.cfi_idx) := io.pbits_update.bits.pbits

    // printf("faubtb update pbits in hit2, cycle: %d, pc: 0x%x, way: %d, cif_idx: %d, pbits: %d\n", debug_cycles.value,
    //   update_pc << 3.U, hitinfo2.way, io.pbits_update.bits.cfi_idx, io.pbits_update.bits.pbits)
  }
  .elsewhen(io.pbits_update.valid && (update_pc === hitinfo3.pc) && hitinfo3.valid){//有效，并且需要更新faubtb
    pbits_valid(hitinfo3.way)(io.pbits_update.bits.cfi_idx) := 1.U
    pbits(hitinfo3.way)(io.pbits_update.bits.cfi_idx) := io.pbits_update.bits.pbits

    // printf("faubtb update pbits in hit3, cycle: %d, pc: 0x%x, way: %d, cif_idx: %d, pbits: %d\n", debug_cycles.value,
    //   update_pc << 3.U, hitinfo3.way, io.pbits_update.bits.cfi_idx, io.pbits_update.bits.pbits)
  }

}

