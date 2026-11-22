//! CPython 3.12 set order for the positive object keys used by `_bands`.
//! Slot order matters when two points tie on x, or a band point enters the ordered dict.

use super::state::InsertionMap;

#[derive(Clone, Copy, Debug)]
enum Slot {
    Empty,
    Deleted,
    Key(i64),
}

#[derive(Clone, Debug)]
pub struct IntSet {
    slots: Vec<Slot>,
    used: usize,
    filled: usize,
}

impl Default for IntSet {
    fn default() -> Self {
        Self {
            slots: vec![Slot::Empty; 8],
            used: 0,
            filled: 0,
        }
    }
}

impl FromIterator<i64> for IntSet {
    fn from_iter<T: IntoIterator<Item = i64>>(keys: T) -> Self {
        let mut set = Self::default();
        for key in keys {
            set.insert(key);
        }
        set
    }
}

impl IntSet {
    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = i64> + '_ {
        self.slots.iter().filter_map(|slot| match *slot {
            Slot::Key(key) => Some(key),
            _ => None,
        })
    }

    fn probe(&self, key: i64) -> (usize, bool) {
        let mask = self.slots.len() - 1;
        let mut perturb = (key as u64 % ((1 << 61) - 1)) as usize;
        let mut index = perturb & mask;
        let mut deleted = None;
        loop {
            let extra = if index + 9 <= mask { 9 } else { 0 };
            for i in index..=index + extra {
                match self.slots[i] {
                    Slot::Empty => return (deleted.unwrap_or(i), false),
                    Slot::Deleted => deleted = Some(i),
                    Slot::Key(existing) if existing == key => return (i, true),
                    Slot::Key(_) => {}
                }
            }
            perturb >>= 5;
            index = (index * 5 + 1 + perturb) & mask;
        }
    }

    pub fn contains(&self, key: &i64) -> bool {
        self.probe(*key).1
    }

    pub fn insert(&mut self, key: i64) {
        let (index, found) = self.probe(key);
        if found {
            return;
        }
        let vacant = matches!(self.slots[index], Slot::Empty);
        self.slots[index] = Slot::Key(key);
        self.used += 1;
        self.filled += usize::from(vacant);
        if vacant && self.filled * 5 >= (self.slots.len() - 1) * 3 {
            self.resize(self.resize_minimum());
        }
    }

    fn resize_minimum(&self) -> usize {
        self.used * if self.used > 50_000 { 2 } else { 4 }
    }

    fn resize(&mut self, min_used: usize) {
        let size = (min_used + 1).next_power_of_two().max(8);
        let previous = std::mem::replace(&mut self.slots, vec![Slot::Empty; size]);
        self.filled = self.used;
        for slot in previous {
            if let Slot::Key(key) = slot {
                let (index, _) = self.probe(key);
                self.slots[index] = slot;
            }
        }
    }

    /// `set.intersection_update(dict)` inserts the matching keys in dict iteration order.
    pub fn intersect(&mut self, keys: impl Iterator<Item = i64>) {
        *self = keys.filter(|key| self.contains(key)).collect();
    }

    /// `set.difference(dict)` uses either a compact copy with deletions or fresh insertions.
    pub fn difference<V>(&self, other: &InsertionMap<i64, V>) -> Self {
        if (self.used >> 2) <= other.len() {
            return self.iter().filter(|key| other.get(key).is_none()).collect();
        }
        let mut copy = Self::default();
        let size = if self.used * 5 >= 7 * 3 {
            (self.used * 2 + 1).next_power_of_two().max(8)
        } else {
            8
        };
        copy.slots.resize(size, Slot::Empty);
        if size == self.slots.len() && self.used == self.filled {
            copy.slots.copy_from_slice(&self.slots);
        } else {
            for key in self.iter() {
                let (index, _) = copy.probe(key);
                copy.slots[index] = Slot::Key(key);
            }
        }
        copy.used = self.used;
        copy.filled = self.used;
        for (key, _) in other.iter() {
            let (index, found) = copy.probe(key);
            if found {
                copy.slots[index] = Slot::Deleted;
                copy.used -= 1;
            }
        }
        if copy.filled - copy.used > (copy.slots.len() - 1) / 4 {
            copy.resize(copy.resize_minimum());
        }
        copy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpython_order_with_collisions_and_deletions() {
        let keys: Vec<_> = (0..28).map(|i| i * 32 + 1).collect();
        let mut set: IntSet = keys.iter().copied().collect();
        let mut known = InsertionMap::default();
        known.insert(keys[5], ());
        known.insert(keys[22], ());
        // CPython 3.12: list(set(keys).difference(dict.fromkeys([keys[5], keys[22]]))).
        assert_eq!(
            set.difference(&known).iter().collect::<Vec<_>>(),
            [
                1, 129, 257, 385, 513, 641, 769, 65, 193, 321, 865, 449, 577, 833, 33, 289, 417, 545, 673, 801, 97,
                225, 353, 481, 609, 737
            ],
        );
        set.intersect(keys[2..10].iter().rev().copied());
        assert_eq!(set.iter().collect::<Vec<_>>(), [289, 161, 193, 225, 257, 129, 97, 65]);
    }
}
