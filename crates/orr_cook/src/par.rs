//! rayon's parallel iterators, run in the calling thread's job ([`crate::job`]): a pool
//! thread enters the job while it runs a piece of one. Cook code imports [`prelude`] in place
//! of `rayon::prelude`; the iterators its sources make carry the job through every adapter.

use rayon::iter::plumbing::{
    Consumer, Folder, Producer, ProducerCallback, UnindexedConsumer, bridge,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::job::{self, Entered, Job};

pub(crate) mod prelude {
    pub(crate) use super::{IntoParIter, ParIter, ParIterMut, ParSlice, ParSliceMut};
    pub(crate) use rayon::iter::{IndexedParallelIterator, ParallelIterator};
}

/// Items of a loop of light items each piece takes at least ([`InJob::light`]).
pub(crate) const LIGHT_MIN: usize = 1 << 16;

/// A parallel iterator whose pieces run in `job`, each at least `min` items.
pub(crate) struct InJob<I> {
    base: I,
    job: Option<Job>,
    min: usize,
}

impl<I> InJob<I> {
    fn new(base: I) -> Self {
        Self {
            base,
            job: job::current(),
            min: 1,
        }
    }

    /// Pieces of at least [`LIGHT_MIN`] items, for items that each take a few nanoseconds: a
    /// shorter loop then runs on the calling thread, where a job thread would otherwise wait
    /// for a busy pool to take it, longer than the loop takes.
    pub(crate) fn light(self) -> Self {
        Self {
            min: LIGHT_MIN,
            ..self
        }
    }
}

impl<I: IndexedParallelIterator> ParallelIterator for InJob<I> {
    type Item = I::Item;

    fn drive_unindexed<C: UnindexedConsumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.base.len())
    }
}

impl<I: IndexedParallelIterator> IndexedParallelIterator for InJob<I> {
    fn len(&self) -> usize {
        self.base.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let (job, min) = (self.job, self.min);
        self.base.with_producer(Callback { callback, job, min })
    }
}

struct Callback<CB> {
    callback: CB,
    job: Option<Job>,
    min: usize,
}

impl<T, CB: ProducerCallback<T>> ProducerCallback<T> for Callback<CB> {
    type Output = CB::Output;

    fn callback<P: Producer<Item = T>>(self, base: P) -> CB::Output {
        self.callback.callback(InJobProducer {
            base,
            job: self.job,
            min: self.min,
        })
    }
}

/// Runs its pieces in `job`: folded, or iterated by adapters that iterate their bases.
struct InJobProducer<P> {
    base: P,
    job: Option<Job>,
    min: usize,
}

impl<P: Producer> Producer for InJobProducer<P> {
    type Item = P::Item;
    type IntoIter = InJobIter<P::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        let entered = job::enter(self.job);
        InJobIter {
            base: self.base.into_iter(),
            _entered: entered,
        }
    }

    fn min_len(&self) -> usize {
        self.base.min_len().max(self.min)
    }

    fn max_len(&self) -> usize {
        self.base.max_len()
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (a, b) = self.base.split_at(index);
        let (job, min) = (self.job, self.min);
        (Self { base: a, job, min }, Self { base: b, job, min })
    }

    fn fold_with<F: Folder<Self::Item>>(self, folder: F) -> F {
        let _entered = job::enter(self.job);
        self.base.fold_with(folder)
    }
}

/// A piece's iterator, in its job until it drops.
struct InJobIter<I> {
    base: I,
    _entered: Entered,
}

impl<I: Iterator> Iterator for InJobIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<I::Item> {
        self.base.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }
}

impl<I: DoubleEndedIterator> DoubleEndedIterator for InJobIter<I> {
    fn next_back(&mut self) -> Option<I::Item> {
        self.base.next_back()
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for InJobIter<I> {}

/// `into_par_iter` in the calling job.
pub(crate) trait IntoParIter: IntoParallelIterator + Sized {
    fn into_par_iter(self) -> InJob<Self::Iter>
    where
        Self::Iter: IndexedParallelIterator,
    {
        InJob::new(IntoParallelIterator::into_par_iter(self))
    }
}

impl<T: IntoParallelIterator> IntoParIter for T {}

/// `par_iter` in the calling job.
pub(crate) trait ParIter<'d> {
    type Iter: IndexedParallelIterator;
    fn par_iter(&'d self) -> InJob<Self::Iter>;
}

impl<'d, T: ?Sized + 'd> ParIter<'d> for T
where
    &'d T: IntoParallelIterator<Iter: IndexedParallelIterator>,
{
    type Iter = <&'d T as IntoParallelIterator>::Iter;

    fn par_iter(&'d self) -> InJob<Self::Iter> {
        InJob::new(IntoParallelIterator::into_par_iter(self))
    }
}

/// `par_iter_mut` in the calling job.
pub(crate) trait ParIterMut<'d> {
    type Iter: IndexedParallelIterator;
    fn par_iter_mut(&'d mut self) -> InJob<Self::Iter>;
}

impl<'d, T: ?Sized + 'd> ParIterMut<'d> for T
where
    &'d mut T: IntoParallelIterator<Iter: IndexedParallelIterator>,
{
    type Iter = <&'d mut T as IntoParallelIterator>::Iter;

    fn par_iter_mut(&'d mut self) -> InJob<Self::Iter> {
        InJob::new(IntoParallelIterator::into_par_iter(self))
    }
}

/// rayon's slice iterators in the calling job.
pub(crate) trait ParSlice<T: Sync> {
    fn par_chunks(&self, size: usize) -> InJob<rayon::slice::Chunks<'_, T>>;
    fn par_windows(&self, size: usize) -> InJob<rayon::slice::Windows<'_, T>>;
}

impl<T: Sync> ParSlice<T> for [T] {
    fn par_chunks(&self, size: usize) -> InJob<rayon::slice::Chunks<'_, T>> {
        InJob::new(ParallelSlice::par_chunks(self, size))
    }

    fn par_windows(&self, size: usize) -> InJob<rayon::slice::Windows<'_, T>> {
        InJob::new(ParallelSlice::par_windows(self, size))
    }
}

/// rayon's mutable slice iterators in the calling job, and its unstable sort, which
/// allocates nothing to charge a job.
pub(crate) trait ParSliceMut<T: Send> {
    fn par_chunks_mut(&mut self, size: usize) -> InJob<rayon::slice::ChunksMut<'_, T>>;
    fn par_sort_unstable(&mut self)
    where
        T: Ord;
}

impl<T: Send> ParSliceMut<T> for [T] {
    fn par_chunks_mut(&mut self, size: usize) -> InJob<rayon::slice::ChunksMut<'_, T>> {
        InJob::new(ParallelSliceMut::par_chunks_mut(self, size))
    }

    fn par_sort_unstable(&mut self)
    where
        T: Ord,
    {
        ParallelSliceMut::par_sort_unstable(self);
    }
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use crate::job::{self, Meter};

    /// Pool work runs in the job of the loop that made it, through adapters that fold their
    /// base and ones that iterate it, nested loops included, and leaves the pool threads in no
    /// job afterwards.
    #[test]
    fn pool_work_runs_in_its_job() {
        let meter = Meter::new();
        let ours = meter.job("a");
        let entered = job::enter(Some(ours));
        let a: Vec<u32> = (0..4096).collect();
        let mut b = vec![0u32; 4096];
        let in_job = || job::current() == Some(ours);
        b.par_iter_mut()
            .zip(a.par_chunks(1))
            .enumerate()
            .for_each(|(i, (x, y))| {
                assert!(in_job());
                *x = y[0] + i as u32;
                let nested: u32 = (0..64u32)
                    .into_par_iter()
                    .map(|k| {
                        assert!(in_job());
                        k
                    })
                    .sum();
                assert_eq!(nested, 2016);
            });
        let kept: Vec<u32> = a
            .par_iter()
            .filter(|&&v| {
                assert!(in_job());
                v % 2 == 0
            })
            .copied()
            .collect();
        assert_eq!(kept.len(), 2048);
        assert_eq!(b[7], 14);
        drop(entered);
        let none = rayon::broadcast(|_| job::current().is_none());
        assert!(none.iter().all(|&n| n), "pool threads left in a job");
    }
}
