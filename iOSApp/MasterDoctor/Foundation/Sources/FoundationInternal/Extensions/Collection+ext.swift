//  Created by Dmitrii Korchagin on 25.11.2025.

import Foundation

public extension Collection {
    /// Возвращает `nil`, если коллекция пуста, в противном случае — саму коллекцию.
    var nilIfEmpty: Self? {
        isEmpty ? nil : self
    }
}
