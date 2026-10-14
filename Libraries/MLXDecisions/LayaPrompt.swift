// Copyright © 2026 Eigen Labs Inc.
// Prompt formatting follows Laya/laya-mlx (Apache-2.0). See NOTICE.
import Foundation
import Tokenizers

struct LayaPreparedQuestion: Sendable {
    let ids: [Int32]
    let markers: [Int32]
    let type: Int32
}

struct LayaPrompt: Sendable {
    let tokenizer: any Tokenizer
    let mask: String
    let maskID: Int
    let clsID: Int
    let sepID: Int
    let padID: Int

    init(directory: URL) async throws {
        let loadedTokenizer = try await AutoTokenizer.from(
            modelFolder: directory.appending(path: "tokenizer"))
        let data = try Data(
            contentsOf: directory.appending(path: "tokenizer/tokenizer_config.json"))
        let config = try DecisionJSON.parse(data)
        func special(_ name: String) throws -> (String, Int) {
            guard let value = config[name], let text = value.string ?? value["content"]?.string,
                let id = loadedTokenizer.convertTokenToId(text)
            else {
                throw LayaError.invalidCheckpoint("Missing tokenizer special token \(name)")
            }
            return (text, id)
        }
        (mask, maskID) = try special("mask_token")
        clsID = try special("cls_token").1
        sepID = try special("sep_token").1
        padID = try special("pad_token").1
        tokenizer = loadedTokenizer
    }

    func prepare(_ request: SystemOneRequest, config: LayaAgentConfiguration) throws
        -> [LayaPreparedQuestion]
    {
        func encode(_ text: String) -> [Int] {
            tokenizer.encode(
                text: text.replacingOccurrences(of: mask, with: " "), addSpecialTokens: false)
        }
        let state = encode(SystemOneRequest.text(request.state))
        return try request.questions.map { question in
            var options = question.options.map { [maskID] + Array(encode(" " + $0).prefix(48)) }
            var budget = config.head_max_len - options.reduce(0) { $0 + $1.count }
            if budget < 16 {
                let per = max(4, (config.head_max_len - 16) / max(1, options.count))
                options = options.map { Array($0.prefix(per)) }
                budget = config.head_max_len - options.reduce(0) { $0 + $1.count }
            }
            let head = encode(question.type + " question: " + question.instructions).prefix(
                max(8, budget))
            var ids = [clsID] + head + [sepID]
            var markers: [Int] = []
            for option in options {
                markers.append(ids.count)
                ids.append(contentsOf: option)
            }
            ids.append(sepID)
            // Never silently drop a decision option or the final separator.
            guard ids.count < config.max_len else {
                throw LayaError.invalidRequest(
                    "Question options exceed the checkpoint token budget")
            }
            ids.append(contentsOf: state.prefix(max(0, config.max_len - ids.count - 1)))
            ids.append(sepID)
            return LayaPreparedQuestion(
                ids: ids.map(Int32.init), markers: markers.map(Int32.init),
                type: Int32(question.typeIndex))
        }
    }
}
