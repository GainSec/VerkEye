/*******************************************************************************
 * debug.c
 *
 * History:
 *    2018/08/22  - [Tao Wu] created
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.	   See the NOTICE file
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
#include <string.h>
#include <sys/stat.h>

#include "cavalry_gen.h"
#include "nnctrl_priv.h"
#include "utils.h"

#define PATH_LENGTH		(256)
#define FILENAME_LENGTH	(512)

#define BLOB_DUMP_DIR	"blob_dump"
#define DVI_DUMP_DIR	"dvi_dump"

static char *char_replace(char *dest, const char *src, size_t size, char olds, char news)
{
	size_t len = 0;
	int i = 0;

	if (src && dest) {
		len = strlen(src);
		strncpy(dest, src, size - 1);

		for (i = 0; i < len && i < size; i++) {
			if (dest[i] == olds) {
				dest[i] = news;
			}
		}
		dest[i] = '\0';
	}

	return dest;
}

int dump_blob_and_dvi(struct net_desc *pnet, const char *path)
{
#define NAME_SIZE (PATH_LENGTH + CAVALRY_IO_DEMNGL_NAME_MAX + 32)

	struct stat status;
	FILE *out = NULL;
	dvi_node_t *node = NULL;
	char dvi_dump_prefix[PATH_LENGTH] = {0};
	char blob_dump_prefix[PATH_LENGTH] = {0};
	char dvi_dump_buffer[NAME_SIZE] = {0};
	char blob_dump_buffer[NAME_SIZE] = {0};
	char filename[CAVALRY_IO_DEMNGL_NAME_MAX] = {0};
	uint32_t i = 0;
	int rval = 0;

	if (stat(path, &status) < 0) {
		perror(path);
		return -1;
	}

	snprintf(dvi_dump_prefix, sizeof(dvi_dump_prefix), "%s/%s", path, DVI_DUMP_DIR);
	snprintf(blob_dump_prefix, sizeof(blob_dump_prefix), "%s/%s", path, BLOB_DUMP_DIR);

	do {
		if (stat(dvi_dump_prefix, &status) == -1) {
			if (mkdir(dvi_dump_prefix, 0755) < 0) {
				perror(dvi_dump_prefix);
				rval = -1;
				break;
			}
		}
		if (stat(blob_dump_prefix, &status) == -1) {
			if (mkdir(blob_dump_prefix, 0755) < 0) {
				perror(blob_dump_prefix);
				rval = -1;
				break;
			}
		}
		if (pnet->reuse_mem == 0) {
			printf("Dump blob dir: %s\n", blob_dump_prefix);
			for (i = 0; i < pnet->blob_num; i++) {
				char_replace(filename, pnet->blob_name[i], sizeof(filename), '/', '_');
				snprintf(blob_dump_buffer, sizeof(blob_dump_buffer),
					"%s/%s.bin", blob_dump_prefix, pnet->blob_name[i]);
				if (pnet->verbose) {
					printf("dump blob id: %u, dvi file: %s, size: %u\n", i,
						blob_dump_buffer, pnet->blob_size[i]);
				}
				out = fopen(blob_dump_buffer, "w+b");
				if (out == NULL) {
					perror(blob_dump_buffer);
					rval = -1;
					break;
				}
				if (fwrite(pnet->virt_addr + pnet->blob_addr[i], 1,
					pnet->blob_size[i], out) != pnet->blob_size[i]) {
					perror(blob_dump_buffer);
					rval = -1;
					break;
				}
				if (out) {
					fclose(out);
					out = NULL;
				}
			}
			if (rval < 0) {
				break;
			}
		}

		printf("Dump dvi dir: %s\n", dvi_dump_prefix);
		node = pnet->dvi_node_list;
		for (i = 0; i < pnet->header.dvi_num; i++) {
			snprintf(dvi_dump_buffer, sizeof(dvi_dump_buffer),
				"%s/%s.dvi", dvi_dump_prefix, node->dvi_desc.dag_name);
			if (pnet->verbose) {
				printf("dump dvi id: %u, dvi file: %s, size: %u\n", i,
					dvi_dump_buffer, node->dvi_desc.dvi_img_size);
			}
			out = fopen(dvi_dump_buffer, "w+b");
			if (out == NULL) {
				perror(dvi_dump_buffer);
				rval = -1;
				break;
			}
			if (fwrite(pnet->virt_addr + node->dvi_dram_addr, 1,
				node->dvi_desc.dvi_img_size, out) != node->dvi_desc.dvi_img_size) {
				perror(dvi_dump_buffer);
				rval = -1;
				break;
			}
			if (out) {
				fclose(out);
				out = NULL;
			}
			node++;
		}
	} while (0);

	if (out) {
		fclose(out);
		out = NULL;
	}

	return rval;
}

static void show_port_info(port_desc_t *port, uint32_t index,
	uint32_t net_phy, uint32_t is_input)
{
	const char *port_type_char[] = {
		"INVALID",
		"INPUT",
		"ITM",
		"ITM_FRM_OUT",
		"OUTPUT"
	};

	printf("%sput_id: %u port_size: %u port_byte_offset: %u bitvector: %u "
		"dim: (P, D, H, W) = ( %d, %d, %d, %d) pitch: %u pitch_offset: %u pitch_bsize: %u "
		"dram_format: %u data_format: (%u, %u, %d, %u) "
		"port_rotate_bit (p, d, v, h, r): (%u, %u, %u, %u, %u) "
		"port_rotate_offset (p, d, v, h, r): (%u, %u, %u, %u, %u) "
		"slice: (seq: %u total: %u offset: %u parent_name: %s) "
		"is_main_input_output: %u port_name: %s (layer_name: %s). "
		"dvi_id: %u port_type: %s dram_size: %u, "
		"port_dram_addr offset: 0x%08x, ",
		(is_input ? "in" : "out"), index, port->io_desc.port_size,
		port->io_desc.port_byte_offset, port->io_desc.port_dim_bitvector,
		port->io_desc.port_dim_p, port->io_desc.port_dim_d,
		port->io_desc.port_dim_h, port->io_desc.port_dim_w,
		port->io_desc.port_pitch, port->io_desc.port_pitch_offset, port->io_desc.port_pitch_bsize,

		port->io_desc.port_dram_format,
		port->io_desc.port_data_sign, port->io_desc.port_data_size,
		port->io_desc.port_data_expoffset, port->io_desc.port_data_expbits,

		port->io_desc.port_pflip, port->io_desc.port_dflip,
		port->io_desc.port_vflip, port->io_desc.port_hflip,
		port->io_desc.port_drotate,
		port->io_desc.port_pflip_bit_offset, port->io_desc.port_dflip_bit_offset,
		port->io_desc.port_vflip_bit_offset, port->io_desc.port_hflip_bit_offset,
		port->io_desc.port_drotate_bit_offset,

		port->io_desc.port_slice_seq, port->io_desc.port_slice_total_num,
		port->io_desc.port_slice_byte_offset, port->io_desc.port_slice_parent_name,
		port->io_desc.port_is_main_io,
		port->io_desc.port_name, port->io_desc.port_demangled_name,

		port->dvi_id, port_type_char[port->port_type], port->port_dram_size,
		port->port_dram_addr);

	if (net_phy) {
		printf("[phys: 0x%08x]\n", net_phy + port->port_dram_addr);
	} else {
		printf("[fd: %u]\n", port->port_dram_fd);
	}
}

void show_dvi_node_list(struct net_desc *pnet)
{
	cavalry_gen_header_t *hdr = NULL;
	dvi_node_t *node = NULL;
	dvi_desc_t *dvi = NULL;
	port_desc_t *port = NULL;
	uint32_t net_phy = 0;
	uint32_t i = 0, j = 0;

	net_phy = pnet->phy_addr;
	hdr = &pnet->header;

	/* Compare the following log to cavalry_info.txt of cavalry_gen */
	printf("\nversion %u.%u.%u (HASH %x) dvi_num: %d\n\n",
		(hdr->version_info & 0xff000000) >> 24, (hdr->version_info & 0x00ff0000) >> 16,
		(hdr->version_info & 0x0000ffff), hdr->version_hash, hdr->dvi_num);

	node = pnet->dvi_node_list;
	for (i = 0; i < pnet->header.dvi_num; i++, node++) {
		dvi = &node->dvi_desc;
		printf("dvi_id: %u dvi_img_vaddr: %u dvi_img_size: %u dvi_dag_vaddr: %u "
			"input_num: %u output_num: %u dvi_pkg_size: %u "
			"dvi_ppv: %u vproc_id: %u dag_name: %s. "
			"dvi_dram_addr offset: 0x%x, ",
			dvi->dvi_id, dvi->dvi_img_vaddr, dvi->dvi_img_size, dvi->dvi_dag_vaddr,
			dvi->input_num, dvi->output_num, dvi->dvi_pkg_size,
			dvi->dvi_ppv,dvi->vproc_id, dvi->dag_name,
			node->dvi_dram_addr);

		if (net_phy) {
			printf("[phys: 0x%08x]\n", net_phy + node->dvi_dram_addr);
		} else {
			printf("[fd: %u]\n", pnet->mem_fd);
		}
	}
	printf("\n");

	node = pnet->dvi_node_list;
	for (i = 0; i < pnet->header.dvi_num; i++, node++) {
		dvi = &node->dvi_desc;
		printf("dvi_id: %u dag_name: %s input_num: %u output_num: %u\n",
			dvi->dvi_id, dvi->dag_name, dvi->input_num, dvi->output_num);

		port = node->in_port;
		for(j = 0; j < dvi->input_num; j++, port++) {
			show_port_info(port, j, pnet->phy_addr, 1);
		}

		port = node->out_port;
		for(j = 0; j < dvi->output_num; j++, port++) {
			show_port_info(port, j, pnet->phy_addr, 0);
		}
		printf("\n");
	}
}
