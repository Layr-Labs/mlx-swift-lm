// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import Testing

@testable import MLXLLM

extension UnitTests {
    /// The paged DSA pool's host bookkeeping (`GLM5NextDSAPageBook`,
    /// `GLM5NextDSAPageTable`): page tables, reference counts, write tokens
    /// and floors (copy on write), rollback, retire/recycle and reservations.
    /// No device: the tables' append plans are checked without the kernels.
    @Suite struct GLM5NextDSAPageBookTests {
        private let lock = NSRecursiveLock()

        private func table(_ book: GLM5NextDSAPageBook) -> GLM5NextDSAPageTable {
            GLM5NextDSAPageTable(book: book, lock: lock)
        }

        @Test func appendsFillPagesInOrderAndRecycleOnlyOnRequest() throws {
            let book = GLM5NextDSAPageBook(pageSize: 4)
            book.grow(to: 8)
            let account = book.openAccount(reservedPages: 6)
            var history: GLM5NextDSAPageTable? = table(book)
            let first = try history!.planAppend(count: 6, account: account)
            #expect(first.clone == nil)
            #expect(history!.pages == [0, 1])
            #expect(first.slots == [0, 1, 2, 3, 4, 5])
            // The tail page is this table's: the next append writes it in place.
            let second = try history!.planAppend(count: 3, account: account)
            #expect(second.clone == nil)
            #expect(second.slots == [6, 7, 8])
            #expect(history!.pages == [0, 1, 2])
            #expect(account.allocatedPages == 3 && book.freePages == 5)
            // Rollback to 5 tokens drops page 2; it is retired, not free.
            history!.truncate(to: 5)
            #expect(history!.pages == [0, 1] && history!.tokenCount == 5)
            #expect(book.retiredPages == 1 && book.freePages == 5 && account.allocatedPages == 3)
            book.recycle()
            #expect(book.retiredPages == 0 && book.freePages == 6 && account.allocatedPages == 2)
            // Writing past the rollback point reuses the tail page in place.
            let third = try history!.planAppend(count: 2, account: account)
            #expect(third.clone == nil && third.slots == [5, 6])
            history = nil
            #expect(book.retiredPages == 2)
            book.recycle()
            #expect(book.freePages == 8 && book.allocatedPages == 0 && account.allocatedPages == 0)
        }

        /// A clone shares every page and takes the write tokens; the original's
        /// holders keep reading below the floor, so the clone writes in place
        /// at and past it, and the original copies its tail before writing.
        @Test func clonesShareAndCopyOnlyAtTheBoundary() throws {
            let book = GLM5NextDSAPageBook(pageSize: 4)
            book.grow(to: 16)
            let account = book.openAccount(reservedPages: 12)
            let source = table(book)
            _ = try source.planAppend(count: 6, account: account)
            let clone = source.clone()
            #expect(clone.pages == source.pages && book.refCount(source.pages[1]) == 2)
            // The clone appends after the shared tokens in the shared tail page.
            let appended = try clone.planAppend(count: 3, account: account)
            #expect(appended.clone == nil && appended.slots == [6, 7, 8])
            // The source has no token now: its own append copies its tail first.
            let diverged = try source.planAppend(count: 1, account: account)
            let copy = try #require(diverged.clone)
            #expect(copy.source == 1 && copy.rows == 2)
            #expect(source.pages[1] == copy.destination && clone.pages[1] == 1)
            #expect(diverged.slots == [copy.destination * 4 + 2])
            #expect(book.refCount(1) == 1 && book.refCount(0) == 2)
        }

        /// A prefix entry keeps the request's table; the request's next append
        /// clones it and writes in place past the entry's tokens, a restore
        /// from the entry copies the shared partial page first, and a write
        /// below what another holder reads (after a rollback) copies too.
        @Test func floorsProtectWhatOtherHoldersRead() throws {
            let book = GLM5NextDSAPageBook(pageSize: 4)
            book.grow(to: 16)
            let account = book.openAccount(reservedPages: 12)
            let stored = table(book)
            _ = try stored.planAppend(count: 5, account: account)
            let boundary = stored.pages[1]
            // The request goes on past the entry's 5 tokens, in place.
            let request = stored.clone()
            let resumed = try request.planAppend(count: 2, account: account)
            #expect(resumed.clone == nil && resumed.slots == [boundary * 4 + 1, boundary * 4 + 2])
            // A restore from the entry holds no token: it copies the shared
            // partial page (one token) and shares the full one.
            let restored = stored.clone()
            let restoredAppend = try restored.planAppend(count: 2, account: account)
            let copy = try #require(restoredAppend.clone)
            #expect(copy.source == boundary && copy.rows == 1)
            #expect(restored.pages[0] == stored.pages[0] && restored.pages[1] != boundary)
            #expect(try restored.planAppend(count: 1, account: account).clone == nil)
            // Rolled back below what the entry reads in page 0, the request copies it.
            request.truncate(to: 4)
            #expect(try request.planAppend(count: 1, account: account).clone == nil)
            request.truncate(to: 3)
            #expect(try request.planAppend(count: 1, account: account).clone?.rows == 3)
            #expect(book.refCount(stored.pages[0]) == 2 && book.refCount(boundary) == 1)
        }

        /// Free pages always cover the open reservations: an account past its
        /// reservation may not take a page another account was promised, and
        /// retired pages count until they are recycled.
        @Test func reservationsAreHonoured() throws {
            let book = GLM5NextDSAPageBook(pageSize: 4)
            book.grow(to: 4)
            let first = book.openAccount(reservedPages: 2)
            let second = book.openAccount(reservedPages: 2)
            #expect(book.requiredPages() == 4 && book.outstandingPages == 4)
            let a = table(book)
            _ = try a.planAppend(count: 8, account: first)
            #expect(first.allocatedPages == 2 && book.outstandingPages == 2)
            #expect(throws: GLM5NextDSAPageBook.Failure.self) {
                try a.planAppend(count: 1, account: first)
            }
            #expect(a.tokenCount == 8 && a.pages.count == 2)
            let b = table(book)
            _ = try b.planAppend(count: 8, account: second)
            #expect(book.freePages == 0)
            // Closing an account hands its live pages to the book as orphans.
            book.close(first)
            #expect(book.orphanPages == 2 && book.reservedPages == 2)
            a.truncate(to: 0)
            #expect(book.orphanPages == 2 && book.retiredPages == 2)
            book.recycle()
            #expect(book.orphanPages == 0 && book.freePages == 2)
            #expect(book.requiredPages(adding: 3) == 5)
        }

        /// A request that grows on demand: its reservation is extended a
        /// granule at a time, and an extension promises pages to that account
        /// as an opening reservation does (another account cannot take them).
        @Test func anAccountGrowsByExtension() throws {
            let book = GLM5NextDSAPageBook(pageSize: 4)
            book.grow(to: 6)
            let growing = book.openAccount(reservedPages: 2)
            let other = book.openAccount(reservedPages: 2)
            let a = table(book)
            _ = try a.planAppend(count: 8, account: growing)
            // Two free pages remain, both promised to `other`.
            #expect(book.freePages == 4 && book.outstandingPages == 2)
            book.extend(growing, by: 2)
            #expect(
                growing.reservedPages == 4 && book.outstandingPages == 4
                    && book.requiredPages() == 6)
            _ = try a.planAppend(count: 8, account: growing)
            #expect(growing.allocatedPages == 4 && a.tokenCount == 16)
            // The growing account's extension is spent; the rest stays `other`'s.
            #expect(throws: GLM5NextDSAPageBook.Failure.self) {
                try a.planAppend(count: 1, account: growing)
            }
            let b = table(book)
            _ = try b.planAppend(count: 8, account: other)
            #expect(book.freePages == 0)
            // A closed account takes no extension.
            book.close(other)
            book.extend(other, by: 3)
            #expect(other.reservedPages == 2 && book.reservedPages == 4)
        }

        @Test func shrinkKeepsEveryLivePage() throws {
            let book = GLM5NextDSAPageBook(pageSize: 4)
            book.grow(to: 8)
            let account = book.openAccount(reservedPages: 8)
            let a = table(book)
            let b = table(book)
            _ = try a.planAppend(count: 8, account: account)
            _ = try b.planAppend(count: 4, account: account)
            #expect(a.pages == [0, 1] && b.pages == [2])
            #expect(book.usedExtent == 3)
            #expect(throws: GLM5NextDSAPageBook.Failure.self) { try book.shrink(to: 2) }
            a.truncate(to: 0)
            book.recycle()
            #expect(book.usedExtent == 3)
            b.truncate(to: 0)
            book.recycle()
            #expect(book.usedExtent == 0)
            book.close(account)
            try book.shrink(to: 0)
            #expect(book.capacityPages == 0 && book.freePages == 0)
            book.grow(to: 4)
            let c = table(book)
            _ = try c.planAppend(count: 1, account: book.openAccount(reservedPages: 1))
            #expect(c.pages == [0])
        }
    }
}
