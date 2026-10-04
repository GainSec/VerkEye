/*******************************************************************************
 * cavalry_mem.c
 *
 * History:
 *    2018/09/18  - [Tao Wu] created
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.	 See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
******************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <cavalry_ioctl.h>
#include <cavalry_mem.h>

#include "mem_priv.h"
#include "mem_ver.h"

static struct cavalry_mem_version G_version = {
	.major = MEM_LIB_MAJOR,
	.minor = MEM_LIB_MINOR,
	.patch = MEM_LIB_PATCH,
	.mod_time = 0x20200428,
	.description = "Cavalry Memory Allocator Library",
};

static struct cavalry_mem_info G_mem_priv;
static unsigned long G_page_size;

#ifndef ROUND_UP
#define ROUND_UP(size, align) (((size) + ((align) - 1)) & ~((align) - 1))
#endif

int cavalry_mem_init(int fd_cav, uint8_t verbose)
{
	struct cavalry_mem_info *priv = &G_mem_priv;
	struct cavalry_mem_version *pver = &G_version;
	long size = 0;

	if (fd_cav < 0) {
		printf("Invalid fd_cavalry: %d param\n", fd_cav);
		return -1;
	}

	if (priv->init_done) {
		printf("Library is inited already, do not do it again\n");
		return -1;
	} else {
		priv->fd_cav = fd_cav;
		priv->verbose = !!verbose;
	}
	if (verbose) {
		printf("%s: %u.%u.%u, mod-time: 0x%x, built-time: %s - %s\n",
			pver->description, pver->major, pver->minor, pver->patch, pver->mod_time,
			__DATE__, __TIME__);
	}

	size = sysconf(_SC_PAGESIZE);
	if (size < 0) {
		perror("sysconf _SC_PAGESIZE");
		return -1;
	}
	G_page_size = size;
	if (priv->verbose) {
		printf("cavalry_mem page size: 0x%lx\n", G_page_size);
	}

	priv->init_done = 1;

	return 0;
}

int cavalry_mem_get_version(struct cavalry_mem_version *ver)
{
	struct cavalry_mem_version *pver = &G_version;

	if (ver == NULL) {
		printf("Version pointer is NULL\n");
		return -1;
	}

	memcpy(ver, pver, sizeof(struct cavalry_mem_version));
	return 0;
}

static int alloc_cache_recycle(unsigned long *psize, unsigned long *pphys,
	void **pvirt, uint8_t cache_en, uint8_t auto_recycle)
{
	struct cavalry_mem_info *priv = &G_mem_priv;
	struct cavalry_mem cv_mem = {0};
	uint8_t *virt = NULL;
	int rval = 0;

	if (!priv->init_done) {
		printf("Library is not inited for alloc\n");
		return -1;
	}
	if ((psize == NULL) || (pphys == NULL) || (pvirt == NULL)) {
		printf("Invalid mem alloc param\n");
		return -1;
	}
	if (*psize == 0) {
		printf("Invalid mem alloc size: %lu\n", *psize);
		return -1;
	}

	cv_mem.length = *psize;
	cv_mem.cache_en = !!cache_en;
	cv_mem.auto_recycle = !!auto_recycle;

	do {
		if (ioctl(priv->fd_cav, CAVALRY_ALLOC_MEM, &cv_mem) < 0) {
			perror("CAVALRY_ALLOC_MEM");
			rval = -1;
			break;
		}

		virt = mmap(NULL, cv_mem.length, PROT_READ | PROT_WRITE, MAP_SHARED,
			priv->fd_cav, cv_mem.offset);
		if (virt == MAP_FAILED) {
			perror("mmap cavalry mem err");
			printf("mem free since mmap err: phys: 0x%08lx, size: 0x%08lx\n",
				cv_mem.offset, cv_mem.length);
			if (ioctl(priv->fd_cav, CAVALRY_FREE_MEM, &cv_mem) < 0) {
				perror("CAVALRY_ALLOC_MEM");
			}
			rval = -1;
			break;
		}
		*pvirt = virt;
		*pphys = cv_mem.offset;
		/* Do not return actual size */
		//*psize = cv_mem.length;

		if (priv->verbose) {
			printf("mem alloc: phys: 0x%08lx, size: 0x%08lx, align_size: 0x%08lx, virt: %p.\n",
				cv_mem.offset, *psize, cv_mem.length, virt);
		}
	}while (0);

	return rval;
}

int cavalry_mem_alloc(unsigned long *psize, unsigned long *pphys,
	void **pvirt, uint8_t cache_en)
{
	return alloc_cache_recycle(psize, pphys, pvirt, cache_en, 1);
}

int cavalry_mem_alloc_persist(unsigned long *psize, unsigned long *pphys,
	void **pvirt, uint8_t cache_en)
{
	return alloc_cache_recycle(psize, pphys, pvirt, cache_en, 0);
}

int cavalry_mem_alloc_mfd(unsigned long size, int *fd,
	void **pvirt, uint8_t cache_en)
{
	struct cavalry_mem_info *priv = &G_mem_priv;
	struct cavalry_mfd_alloc cv_mem = {0};
	uint8_t *virt = NULL;
	int rval = 0;

	if (!priv->init_done) {
		printf("Library is not inited for alloc\n");
		return -1;
	}

	cv_mem.length = size;
	cv_mem.cache_en = !!cache_en;

	do {
		if (ioctl(priv->fd_cav, CAVALRY_ALLOC_MEMFD, &cv_mem) < 0) {
			perror("CAVALRY_ALLOC_MEMFD");
			rval = -1;
			break;
		}

		virt = mmap(NULL, cv_mem.length, PROT_READ | PROT_WRITE, MAP_SHARED,
			cv_mem.fd, 0);
		if (virt == MAP_FAILED) {
			perror("mmap cavalry memfd err");
			printf("mem free since mmapfd err: fd: %u, size: 0x%08lx\n",
				cv_mem.fd, cv_mem.length);
			close(cv_mem.fd);
			rval = -1;
			break;
		}
		*pvirt = virt;
		*fd = cv_mem.fd;

		if (priv->verbose) {
			printf("mem alloc: fd: %u, size: 0x%08lx, virt: %p.\n",
				cv_mem.fd, cv_mem.length, virt);
		}
	} while (0);

	return rval;
}

int cavalry_mem_free(unsigned long size, unsigned long phys, void *virt)
{
	struct cavalry_mem_info *priv = &G_mem_priv;
	struct cavalry_mem cv_mem = {0};
	unsigned long align_size = 0;
	int rval = 0;

	if (!priv->init_done) {
		printf("Library is not inited for alloc\n");
		return -1;
	}
	if ((size == 0) || (phys == 0) || (virt == NULL)) {
		printf("Invalid mem free size, phys, virt param\n");
		return -1;
	}

	align_size = ROUND_UP(size, G_page_size);
	if (munmap(virt, align_size) < 0) {
		perror("munmap cavalry mem err");
		rval = -1;
	}

	cv_mem.offset = phys;
	if (ioctl(priv->fd_cav, CAVALRY_FREE_MEM, &cv_mem) < 0) {
		perror("CAVALRY_FREE_MEM");
		rval = -1;
	}

	if (priv->verbose) {
		printf("mem free: phys: 0x%08lx, size: 0x%08lx, align_size: 0x%08lx, virt: %p.\n",
			cv_mem.offset, size, align_size, virt);
	}

	return rval;
}

int cavalry_mem_free_mfd(unsigned long size, int fd, void *virt)
{
	struct cavalry_mem_info *priv = &G_mem_priv;
	int rval = 0;

	if (!priv->init_done) {
		printf("Library is not inited for alloc\n");
		return -1;
	}
	if ((size == 0) || (fd < 0) || (virt == NULL)) {
		printf("Invalid mem free size, fd, virt param\n");
		return -1;
	}

	if (munmap(virt, size) < 0) {
		perror("munmap cavalry mem err");
		rval = -1;
	}
	close(fd);

	if (priv->verbose) {
		printf("mem free: fd: %u, size: 0x%08lx, virt: %p.\n", fd, size, virt);
	}

	return rval;
}

int cavalry_mem_sync_cache(unsigned long size, unsigned long phys,
	uint8_t clean, uint8_t invalid)
{
	struct cavalry_mem_info *priv = &G_mem_priv;
	struct cavalry_cache_mem cache = {0};
	int rval = 0;

	if (!priv->init_done) {
		printf("Library is not inited for alloc\n");
		return -1;
	}
	if ((size == 0) || (phys == 0)) {
		printf("Invalid sync cache size and phys param\n");
		return -1;
	}

	cache.offset = phys;
	cache.length = size;
	cache.clean = !!clean;
	cache.invalid = !!invalid;
	if (ioctl(priv->fd_cav, CAVALRY_SYNC_CACHE_MEM, &cache) < 0) {
		perror("CAVALRY_SYNC_CACHE_MEM");
		rval = -1;
	}

	if (priv->verbose) {
		printf("mem sync (%u, %u): phys: 0x%08lx, size: 0x%08lx\n",
			cache.clean, cache.invalid, cache.offset, cache.length);
	}

	return rval;
}

int cavalry_mem_sync_cache_mfd(unsigned long size, unsigned long offset, int fd,
	uint8_t clean, uint8_t invalid)
{
	struct cavalry_mem_info *priv = &G_mem_priv;
	struct cavalry_mfd_cache cache = {0};
	int rval = 0;

	if (!priv->init_done) {
		printf("Library is not inited for alloc\n");
		return -1;
	}
	if ((size == 0) || (fd < 0)) {
		printf("Invalid sync cache size and fd param\n");
		return -1;
	}

	cache.fd = fd;
	cache.offset = offset;
	cache.length = size;
	cache.clean = !!clean;
	cache.invalid = !!invalid;
	if (ioctl(priv->fd_cav, CAVALRY_SYNC_CACHE_MEMFD, &cache) < 0) {
		perror("CAVALRY_SYNC_CACHE_MEMFD");
		rval = -1;
	}

	if (priv->verbose) {
		printf("mem sync (%u, %u): fd: %u, offset: 0x%08lx, size: 0x%08lx\n",
			cache.clean, cache.invalid, cache.fd, cache.offset, cache.length);
	}

	return rval;
}

void cavalry_mem_exit(void)
{
	struct cavalry_mem_info *priv = &G_mem_priv;

	priv->fd_cav = -1;
	priv->init_done = 0;
}
