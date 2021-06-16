//******************************************************************************
// Copyright (c) 2017 - 2019, The Regents of the University of California (Regents).
// All Rights Reserved. See LICENSE and LICENSE.SiFive for license details.
//------------------------------------------------------------------------------

package boom.ifu

import chisel3._
import chisel3.util._
import chisel3.core.{withReset}
import chisel3.internal.sourceinfo.{SourceInfo}

import freechips.rocketchip.config._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.tile._
import freechips.rocketchip.util._
import freechips.rocketchip.util.property._
import freechips.rocketchip.diplomaticobjectmodel.logicaltree.{ICacheLogicalTreeNode}

import boom.common._
import boom.exu.{CommitExceptionSignals, BranchDecode, BrUpdateInfo}
import boom.util._

class BoomRAS(implicit p: Parameters) extends BoomModule()(p)
{
  val io = IO(new Bundle {
    val read_idx   = Input(UInt(log2Ceil(nRasEntries).W))
    val read_addr  = Output(UInt(vaddrBitsExtended.W))

    val write_valid = Input(Bool())
    val write_idx   = Input(UInt(log2Ceil(nRasEntries).W))
    val write_addr  = Input(UInt(vaddrBitsExtended.W))

    //chw: for pbits
    val write_pbits     = Input(UInt(2.W))
    val write_pbits_val = Input(Bool())
    val read_pbits      = Output(UInt(2.W))
    val read_pbits_val  = Output(Bool())
  })
  val ras = Reg(Vec(nRasEntries, UInt(vaddrBitsExtended.W)))

  io.read_addr := Mux(RegNext(io.write_valid && io.write_idx === io.read_idx),
    RegNext(io.write_addr),
    RegNext(ras(io.read_idx)))

  when (io.write_valid) {
    ras(io.write_idx) := io.write_addr
  }

  //chw: for pbits
  val pbits = Reg(Vec(nRasEntries, UInt(2.W)))
  val pbits_val = Reg(Vec(nRasEntries, Bool()))
  when (io.write_valid) {
    pbits(io.write_idx)     := io.write_pbits
    pbits_val(io.write_idx) := io.write_pbits_val
  }

  io.read_pbits := Mux(RegNext(io.write_valid && io.write_idx === io.read_idx),
    RegNext(io.write_pbits),
    RegNext(pbits(io.read_idx)))

  io.read_pbits_val := Mux(RegNext(io.write_valid && io.write_idx === io.read_idx),
    RegNext(io.write_pbits_val),
    RegNext(pbits_val(io.read_idx)))
}
