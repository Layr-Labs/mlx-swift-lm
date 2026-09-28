import MLX
import XCTest
@testable import MLXLLM
@testable import MLXVLM

final class MiMoV26ResidualVectorQuantizerTests: XCTestCase {
    private func table(_ values: [Float], width: Int = 8) -> MLXArray {
        MLXArray(values.flatMap { [$0]+Array(repeating:Float(0),count:width-1) }).reshaped(values.count,width)
    }
    func testSequentialResidualAndFirstIndexTie() throws {
        try mimoAudioInputRequireNative()
        let c = try MiMoV26AudioInputConfiguration.fixture(bins:[2,2])
        let q = try MiMoV26ResidualVectorQuantizer(configuration:c,sourceTables:[table([0,4]),table([0,4])])
        let features = table([5,2])
        let out = try q.quantize(features:features,frameCounts:[2],tileFrames:1)
        eval(out.codes,out.allFinite)
        XCTAssertTrue(out.allFinite.item(Bool.self))
        XCTAssertEqual(out.codes.asArray(Int32.self),[1,0,0,0],"stage2 must see residual; equal distances select first index")
    }
    func testSourceFloat32TablesAreRoundedThroughBF16BeforeRuntimeFloat32() throws {
        try mimoAudioInputRequireNative()
        let c = try MiMoV26AudioInputConfiguration.fixture(bins:[2])
        let source = table([1.003,1.001])
        let q = try MiMoV26ResidualVectorQuantizer(configuration:c,sourceTables:[source])
        let out = try q.quantize(features:table([1]),frameCounts:[1],tileFrames:1)
        eval([out.codes]+q.materializationRoots)
        XCTAssertEqual(out.codes.asArray(Int32.self),[0])
        XCTAssertEqual(source.asArray(Float.self)[0],1.003)
        XCTAssertEqual(q.materializationRoots[0].dtype,.float32)
        XCTAssertEqual(q.materializationRoots[0].asArray(Float.self)[0],1)
        XCTAssertLessThan(abs(Float(1.001)-1),abs(Float(1.003)-1),"unrounded source would choose index1")
    }
    func testTiledAndUntiledCodesAndInvalidLengthDTypeTables() throws {
        try mimoAudioInputRequireNative()
        let c = try MiMoV26AudioInputConfiguration.fixture(bins:[2,3])
        XCTAssertThrowsError(try MiMoV26ResidualVectorQuantizer(configuration:c,sourceTables:[table([0,1])]))
        XCTAssertThrowsError(try MiMoV26ResidualVectorQuantizer(configuration:c,sourceTables:[table([0,1]).asType(.bfloat16),table([0,1,2])]))
        let q = try MiMoV26ResidualVectorQuantizer(configuration:c,sourceTables:[table([0,4]),table([-1,0,1])])
        let x = table([0,1,2,3,4,5,6])
        let a = try q.quantize(features:x,frameCounts:[3,4],tileFrames:2), b = try q.quantize(features:x,frameCounts:[3,4],tileFrames:7)
        eval(a.codes,b.codes);XCTAssertEqual(a.codes.asArray(Int32.self),b.codes.asArray(Int32.self))
        XCTAssertThrowsError(try q.quantize(features:x,frameCounts:[6],tileFrames:2))
        XCTAssertThrowsError(try q.quantize(features:x,frameCounts:[7],tileFrames:0))
        XCTAssertThrowsError(try q.quantize(features:x,frameCounts:[7],tileFrames:2,isCancelled:{true}))
    }
    func testNonfiniteRowsUseSafeIndexButCannotPublish() throws {
        try mimoAudioInputRequireNative()
        let c = try MiMoV26AudioInputConfiguration.fixture(bins:[2])
        let q = try MiMoV26ResidualVectorQuantizer(configuration:c,sourceTables:[table([0,4])])
        let x = table([.nan,.infinity,Float.greatestFiniteMagnitude])
        let out = try q.quantize(features:x,frameCounts:[3],tileFrames:2)
        eval(out.codes,out.allFinite)
        XCTAssertFalse(out.allFinite.item(Bool.self))
        XCTAssertTrue(out.codes.asArray(Int32.self).allSatisfy { $0 >= 0 && $0 < 2 })
    }
}
